"""gripper-mask CLI: `mask` generates gripper masks, `qa` builds and serves the review gallery."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .files import enumerate_images
from .pipeline import PipelineConfig, process_image

EXIT_OK = 0
EXIT_WARNINGS = 1  # ran, but some files were skipped or produced empty masks
EXIT_USAGE = 2


def hex_color(value: str) -> tuple[int, int, int]:
    s = value.lstrip("#")
    if len(s) != 6 or any(c not in "0123456789abcdefABCDEF" for c in s):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an RRGGBB hex color (e.g. '#ff6600')"
        )
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def roi_box(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be 'x0,y0,x1,y1' in pixels")
    try:
        x0, y0, x1, y1 = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc
    if x0 >= x1 or y0 >= y1:
        raise argparse.ArgumentTypeError("ROI must satisfy x0 < x1 and y0 < y1")
    return x0, y0, x1, y1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gripper-mask",
        description="Mask out the SO-101 gripper in wrist-cam images via HSV color thresholding.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    mask = sub.add_parser("mask", help="generate mask PNGs for a directory of images")
    mask.add_argument("in_dir", type=Path, help="directory of source images")
    mask.add_argument("out_dir", type=Path, help="directory to write mask PNGs")
    mask.add_argument(
        "--color",
        type=hex_color,
        required=True,
        help="gripper base color as RRGGBB hex, eyedropped from an actual captured frame",
    )
    tuning = mask.add_argument_group("tuning")
    tuning.add_argument("--hue-tol", type=int, default=8, help="hue window half-width, OpenCV 0-179 units (default 8)")
    tuning.add_argument("--sat-min", type=int, default=100, help="minimum saturation 0-255 (default 100)")
    tuning.add_argument("--val-min", type=int, default=60, help="minimum value/brightness 0-255 (default 60)")
    tuning.add_argument("--close-kernel", type=int, default=5, help="morphological close kernel size (default 5)")
    tuning.add_argument("--close-iters", type=int, default=2, help="morphological close iterations (default 2)")
    tuning.add_argument("--dilate", type=int, default=3, help="dilation radius in pixels (default 3)")
    tuning.add_argument("--second-ratio", type=float, default=0.15, help="keep 2nd component when its area >= ratio x largest (default 0.15)")
    tuning.add_argument("--min-area", type=float, default=0.005, help="minimum component area as fraction of frame pixels (default 0.005)")
    tuning.add_argument("--core-sat", type=int, default=165, help="core-score saturation floor for component ranking (default 165)")
    tuning.add_argument("--core-val", type=int, default=175, help="core-score value floor for component ranking (default 175)")
    tuning.add_argument("--roi", type=roi_box, default=None, help="restrict component search to 'x0,y0,x1,y1' pixels (default: full frame)")
    tuning.add_argument("--no-edge-prior", action="store_true", help="disable the bottom-edge spatial preference")
    mask.add_argument("--skip-existing", action="store_true", help="skip images whose mask already exists (default: overwrite)")

    qa = sub.add_parser("qa", help="build and serve the QA gallery for generated masks")
    qa.add_argument("images_dir", type=Path, help="directory of source images")
    qa.add_argument("out_dir", type=Path, help="mask directory (gallery is built in <out_dir>/qa/)")
    qa.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; use 0.0.0.0 for LAN viewing)")
    qa.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    qa.add_argument("--no-serve", action="store_true", help="generate the gallery without starting the server")
    return parser


def cmd_mask(args: argparse.Namespace) -> int:
    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir

    if not in_dir.is_dir():
        print(f"error: input directory not found: {in_dir}", file=sys.stderr)
        return EXIT_USAGE
    images = enumerate_images(in_dir)
    if not images:
        print(f"error: no images (.jpg/.jpeg/.png) in {in_dir}", file=sys.stderr)
        return EXIT_USAGE
    if out_dir.resolve() == in_dir.resolve():
        print("error: out_dir must differ from in_dir", file=sys.stderr)
        return EXIT_USAGE
    stem_counts = Counter(p.stem for p in images)
    collisions = sorted(stem for stem, n in stem_counts.items() if n > 1)
    if collisions:
        print(
            "error: basename collisions would overwrite masks: " + ", ".join(collisions),
            file=sys.stderr,
        )
        return EXIT_USAGE

    import cv2  # deferred: keeps `--help` fast

    cfg = PipelineConfig(
        color=args.color,
        hue_tol=args.hue_tol,
        sat_min=args.sat_min,
        val_min=args.val_min,
        close_kernel=args.close_kernel,
        close_iters=args.close_iters,
        dilate=args.dilate,
        second_ratio=args.second_ratio,
        min_area=args.min_area,
        roi=args.roi,
        edge_prior=not args.no_edge_prior,
        core_sat=args.core_sat,
        core_val=args.core_val,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    processed = empty = skipped = existing = 0
    for src in images:
        dst = out_dir / f"{src.stem}.png"
        if args.skip_existing and dst.exists():
            existing += 1
            continue
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"warning: unreadable image skipped: {src.name}", file=sys.stderr)
            skipped += 1
            continue
        mask, is_empty = process_image(bgr, cfg)
        if is_empty:
            print(f"warning: no gripper detected (all-white mask): {src.name}", file=sys.stderr)
            empty += 1
        cv2.imwrite(str(dst), mask)
        processed += 1

    summary = f"processed {processed}, empty {empty}, skipped {skipped}"
    if existing:
        summary += f", skipped-existing {existing}"
    print(summary)
    return EXIT_WARNINGS if (empty or skipped) else EXIT_OK


def cmd_qa(args: argparse.Namespace) -> int:
    from . import gallery  # deferred: keeps `--help` fast

    if not args.images_dir.is_dir():
        print(f"error: images directory not found: {args.images_dir}", file=sys.stderr)
        return EXIT_USAGE
    if not args.out_dir.is_dir():
        print(f"error: mask directory not found: {args.out_dir}", file=sys.stderr)
        return EXIT_USAGE

    qa_dir, stats = gallery.build_gallery(args.images_dir, args.out_dir)
    print(
        f"gallery: {stats.total} rows ({stats.empty} empty-mask, "
        f"{stats.missing_mask} missing masks, {stats.stale_mask} stale masks) -> {qa_dir}"
    )
    if args.no_serve:
        return EXIT_OK
    return gallery.serve(qa_dir, args.host, args.port)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "mask":
        return cmd_mask(args)
    return cmd_qa(args)


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
