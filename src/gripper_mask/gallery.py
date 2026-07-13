"""QA gallery: self-contained static bundle in <out_dir>/qa/ plus a stdlib server.

The bundle copies originals and masks in because http.server cannot serve files
outside its root; everything the page references lives under qa/.
"""

from __future__ import annotations

import hashlib
import html
import shutil
import socketserver
import sys
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .files import enumerate_images, natural_key

TINT_BGR = (255, 0, 255)  # magenta wash over masked pixels; distinct from gripper orange
TINT_ALPHA = 0.45


@dataclass
class GalleryStats:
    total: int = 0
    empty: int = 0
    missing_mask: int = 0
    stale_mask: int = 0


@dataclass
class Row:
    stem: str
    image_name: str | None  # None for a stale mask with no source image
    empty: bool = False
    missing_mask: bool = False


def _overlay(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Original with a translucent tint over masked (0) pixels — bleed stays visible."""
    out = bgr.copy()
    region = mask == 0
    tint = np.empty_like(bgr)
    tint[:] = TINT_BGR
    blended = cv2.addWeighted(bgr, 1.0 - TINT_ALPHA, tint, TINT_ALPHA, 0.0)
    out[region] = blended[region]
    return out


def build_gallery(images_dir: Path, out_dir: Path) -> tuple[Path, GalleryStats]:
    qa_dir = out_dir / "qa"
    shutil.rmtree(qa_dir, ignore_errors=True)
    for sub in ("originals", "masks", "overlays"):
        (qa_dir / sub).mkdir(parents=True)

    images = {p.stem: p for p in enumerate_images(images_dir)}
    masks = {
        p.stem: p for p in out_dir.glob("*.png") if p.is_file()
    }

    stats = GalleryStats()
    rows: list[Row] = []

    for stem in sorted(images.keys() | masks.keys(), key=natural_key):
        src = images.get(stem)
        mask_path = masks.get(stem)
        row = Row(stem=stem, image_name=src.name if src else None)

        if src is not None:
            shutil.copy2(src, qa_dir / "originals" / src.name)
        if mask_path is not None:
            shutil.copy2(mask_path, qa_dir / "masks" / mask_path.name)

        if src is None:
            row.missing_mask = False
            stats.stale_mask += 1
        elif mask_path is None:
            row.missing_mask = True
            stats.missing_mask += 1
        else:
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if bgr is not None and mask is not None and bgr.shape[:2] == mask.shape[:2]:
                cv2.imwrite(str(qa_dir / "overlays" / f"{stem}.jpg"), _overlay(bgr, mask))
                row.empty = not np.any(mask == 0)
                if row.empty:
                    stats.empty += 1
            else:
                row.missing_mask = True
                stats.missing_mask += 1

        rows.append(row)

    stats.total = len(rows)
    dataset_id = hashlib.sha1(str(out_dir.resolve()).encode()).hexdigest()[:12]
    (qa_dir / "index.html").write_text(_render_html(rows, stats, dataset_id), encoding="utf-8")
    return qa_dir, stats


def _render_row(row: Row) -> str:
    stem = html.escape(row.stem)
    classes = ["row"]
    if row.empty:
        classes.append("empty")
    if row.image_name is None:
        img_cell = '<figure class="missing">source image missing (stale mask)</figure>'
        overlay_cell = '<figure class="missing">no overlay</figure>'
    else:
        name = html.escape(row.image_name)
        img_cell = f'<figure><img loading="lazy" src="originals/{name}"><figcaption>original</figcaption></figure>'
        overlay_cell = (
            f'<figure><img loading="lazy" src="overlays/{stem}.jpg"><figcaption>overlay</figcaption></figure>'
            if not row.missing_mask
            else '<figure class="missing">no overlay</figure>'
        )
    mask_cell = (
        f'<figure><img loading="lazy" src="masks/{stem}.png"><figcaption>mask</figcaption></figure>'
        if not row.missing_mask
        else '<figure class="missing">no mask generated</figure>'
    )
    badge = '<span class="badge">EMPTY MASK</span>' if row.empty else ""
    filename = html.escape(row.image_name or f"{row.stem}.png")
    return f"""
<section class="{' '.join(classes)}" id="row-{stem}" data-file="{filename}">
  <header><label><input type="checkbox" class="flag"> flag</label> <strong>{filename}</strong> {badge}</header>
  <div class="cells">{img_cell}{mask_cell}{overlay_cell}</div>
</section>"""


def _render_html(rows: list[Row], stats: GalleryStats, dataset_id: str) -> str:
    body = "\n".join(_render_row(r) for r in rows)
    notes = []
    if stats.missing_mask:
        notes.append(f"{stats.missing_mask} image(s) have no mask")
    if stats.stale_mask:
        notes.append(f"{stats.stale_mask} stale mask(s) have no source image")
    note_html = f'<span class="note">{html.escape("; ".join(notes))}</span>' if notes else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>gripper-mask QA</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }}
  #topbar {{ position: sticky; top: 0; background: #222; padding: 8px 12px; display: flex;
             gap: 12px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid #444; z-index: 10; }}
  #topbar .note {{ color: #fa0; }}
  button {{ padding: 4px 10px; }}
  button:disabled {{ opacity: 0.4; }}
  textarea {{ width: 320px; height: 2.4em; background: #181818; color: #eee; }}
  .row {{ padding: 10px 12px; border-bottom: 1px solid #333; }}
  .row.empty header strong {{ color: #f66; }}
  .row.flagged {{ background: #2a1a1a; }}
  .badge {{ background: #a00; color: #fff; padding: 1px 6px; border-radius: 3px; font-size: 0.8em; }}
  .cells {{ display: flex; gap: 8px; margin-top: 6px; }}
  .cells img {{ max-width: 31vw; height: auto; display: block; }}
  figure {{ margin: 0; }}
  figcaption {{ font-size: 0.75em; color: #999; text-align: center; }}
  figure.missing {{ display: flex; align-items: center; justify-content: center; min-width: 200px;
                    color: #f66; border: 1px dashed #a00; font-size: 0.85em; }}
  #copied {{ color: #6f6; visibility: hidden; }}
</style>
</head>
<body>
<div id="topbar">
  <span id="counts"></span>
  {note_html}
  <button id="next-empty">next empty</button>
  <button id="next-flagged">next flagged</button>
  <textarea id="flaglist" readonly title="flagged filenames"></textarea>
  <button id="copy" disabled>copy flagged list</button>
  <span id="copied"></span>
</div>
{body}
<script>
const DATASET_ID = "{dataset_id}";
const TOTAL = {stats.total}, EMPTY = {stats.empty};
const rows = Array.from(document.querySelectorAll('.row'));
const flaglist = document.getElementById('flaglist');
const copyBtn = document.getElementById('copy');
const copied = document.getElementById('copied');

function key(row) {{ return DATASET_ID + ':' + row.dataset.file; }}

rows.forEach(row => {{
  const box = row.querySelector('.flag');
  const stored = localStorage.getItem(key(row));
  // Empty-mask rows start pre-checked (system-detected failure); an explicit
  // user un-check is remembered via localStorage.
  box.checked = stored === null ? row.classList.contains('empty') : stored === '1';
  row.classList.toggle('flagged', box.checked);
  box.addEventListener('change', () => {{
    localStorage.setItem(key(row), box.checked ? '1' : '0');
    row.classList.toggle('flagged', box.checked);
    refresh();
  }});
}});

function flaggedFiles() {{
  return rows.filter(r => r.querySelector('.flag').checked).map(r => r.dataset.file);
}}

function refresh() {{
  const flagged = flaggedFiles();
  flaglist.value = flagged.join('\\n');
  copyBtn.disabled = flagged.length === 0;
  document.getElementById('counts').textContent =
    TOTAL + ' frames / ' + EMPTY + ' empty / ' + flagged.length + ' flagged';
}}

function jumper(selector) {{
  let i = -1;
  return () => {{
    const hits = rows.filter(r => r.matches(selector));
    if (!hits.length) return;
    i = (i + 1) % hits.length;
    hits[i].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
  }};
}}
document.getElementById('next-empty').addEventListener('click', jumper('.empty'));
document.getElementById('next-flagged').addEventListener('click', jumper('.flagged'));

copyBtn.addEventListener('click', () => {{
  const flagged = flaggedFiles();
  flaglist.focus();
  flaglist.select();
  // Clipboard API needs a secure context; LAN viewing is plain http, so the
  // pre-selected textarea is the always-works path and clipboard is a bonus.
  if (navigator.clipboard) navigator.clipboard.writeText(flaglist.value).catch(() => {{}});
  else document.execCommand('copy');
  copied.textContent = 'Copied ' + flagged.length + ' filenames';
  copied.style.visibility = 'visible';
  setTimeout(() => {{ copied.style.visibility = 'hidden'; }}, 2000);
}});

refresh();
</script>
</body>
</html>
"""


def serve(qa_dir: Path, host: str, port: int) -> int:
    handler = partial(SimpleHTTPRequestHandler, directory=str(qa_dir))
    socketserver.TCPServer.allow_reuse_address = True
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        print(f"error: cannot bind {host}:{port} ({exc.strerror})", file=sys.stderr)
        return 2
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"serving QA gallery at http://{shown_host}:{port}/ (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
