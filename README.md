# SO-101-gripper-mask

Processes a directory of images and produces a directory of mask images which mask out the gripper for an SO-101 arm with a wrist webcam.

Masks are single-channel PNGs at source resolution with the same basename as the source image: **0 (black) = gripper / masked area, 255 (white) = keep**. If a downstream tool resizes masks, use nearest-neighbor interpolation only — bilinear resizing creates gray boundary values.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+ (uv will fetch Python if needed).

```bash
uv sync
```

## Generate masks

```bash
uv run gripper-mask mask <images_dir> <masks_dir> --color '#fe814c'
```

`--color` is the gripper's color as RRGGBB hex. **Eyedrop it from an actual captured frame** (open a frame in any image viewer and sample a bright spot on the gripper) — do not use the filament's nominal color, because the camera's white balance shifts the rendered hue.

Existing masks are overwritten by default, so tuning iterations take effect without extra flags; pass `--skip-existing` to resume an interrupted run instead.

A one-line summary prints at the end (`processed N, empty N, skipped N`). Frames where no gripper was detected get an all-white mask and a warning rather than a missing file, so downstream basename pairing never breaks.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | every file processed cleanly |
| 1 | ran, but some files were skipped (unreadable) or produced empty masks |
| 2 | usage or environment error (bad flag, missing directory, basename collision) |

### How selection works

Thresholding happens in HSV (hue window + saturation/value floors), but component *selection* happens in **Lab chroma** — the measure of color vibrancy. The gripper is vibrant orange (chroma 59–83 on the reference capture); the warm wall, wood, and corks share its hue but are dull (chroma 23–50). Every thresholded component is kept when at least `--core-frac` of its pixels are vivid (chroma ≥ `--core-chroma`); a component that fails the gate but contains gripper-sized vivid sub-regions (a finger visually touching the wall merges into one blob with it) has those sub-regions cut out and kept. All passing components are kept — a motion-blurred finger can fragment into several blobs, and the tool is biased to never miss gripper pixels.

The final mask is grown outward by `--grow` pixels (default 5) and every enclosed hole is filled, so masks are always solid regions — no pinholes to corrupt gaussian-splat training. Over-masking slightly beyond the gripper is by design.

### Tuning flags

All pipeline parameters are flags with defaults calibrated against the shakedown reference capture: `--hue-tol 8`, `--sat-min 100`, `--val-min 60`, `--close-kernel 5`, `--close-iters 2`, `--dilate 3`, `--min-area 0.005`, `--core-chroma 60`, `--core-frac 0.35`, `--grow 5`, `--roi x0,y0,x1,y1`. See `uv run gripper-mask mask --help`.

## QA gallery

```bash
uv run gripper-mask qa <images_dir> <masks_dir>
```

Builds a self-contained static gallery in `<masks_dir>/qa/` (regenerated from scratch on every run, so it never shows stale masks) and serves it at `http://127.0.0.1:8000/`. Each row shows the original, the mask, and a tinted overlay — the translucent tint makes threshold misses and dilation bleed easy to spot.

- The Jetson is usually headless: pass `--host 0.0.0.0` and browse from another machine at `http://<jetson-ip>:8000/`. `--port` changes the port; `--no-serve` generates without serving.
- Empty-mask rows are highlighted and their flag checkbox starts pre-checked.
- Flag bad frames with the per-row checkbox (persisted in your browser, scoped to this mask directory); the header shows counts and next-empty / next-flagged jump links; the textarea + copy button export the flagged filename list (works over plain-http LAN viewing).

## Known limitation

Scene objects that are as vividly orange as the gripper itself (chroma near or above `--core-chroma`) will be masked along with it — the tool deliberately prefers over-masking to ever missing gripper pixels. Dull warm surfaces (wood, walls, corks) are excluded by the chroma gate even when they share the gripper's hue. Use the QA gallery to review; raise `--core-frac` or `--core-chroma` if a vivid object keeps sneaking in.
