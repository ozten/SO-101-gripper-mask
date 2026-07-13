"""Per-image mask pipeline: threshold, morphology, chroma-core selection, rendering.

Mask convention: single-channel uint8, 0 = gripper (masked area), 255 = keep.
Resize masks with nearest-neighbor only; bilinear creates gray boundary values.

Selection is biased to never miss gripper pixels: every component that passes the
chroma-core gate is kept (over-masking a vivid orange scene object is acceptable;
losing a finger is not), the final mask is grown outward, and enclosed holes are
filled so the mask is always a set of solid regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PipelineConfig:
    color: tuple[int, int, int]  # gripper base color, RGB
    hue_tol: int = 8  # OpenCV hue units (0-179 scale)
    sat_min: int = 100
    val_min: int = 60
    close_kernel: int = 5
    close_iters: int = 2
    dilate: int = 3
    min_area: float = 0.005  # fraction of frame pixels
    roi: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 (pixels)
    core_chroma: int = 60  # Lab chroma floor defining "vividly orange" core pixels
    core_frac: float = 0.35  # component kept when this fraction of it is core
    grow: int = 5  # grow the final mask outward by this many pixels


def rgb_to_hsv(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert an RGB color to OpenCV HSV (H 0-179, S/V 0-255)."""
    r, g, b = color
    px = np.uint8([[[b, g, r]]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    return int(h), int(s), int(v)


def hue_windows(hue: int, tol: int) -> list[tuple[int, int]]:
    """Hue window(s) around `hue`, wrapping modulo 180 past either end."""
    if tol >= 90:
        return [(0, 179)]
    lo = (hue - tol) % 180
    hi = (hue + tol) % 180
    if lo <= hi:
        return [(lo, hi)]
    return [(0, hi), (lo, 179)]


def threshold(hsv: np.ndarray, hue: int, tol: int, sat_min: int, val_min: int) -> np.ndarray:
    """Binary mask (255 = in-range) via dual-range hue windows plus S/V floors."""
    out = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in hue_windows(hue, tol):
        out |= cv2.inRange(hsv, (lo, sat_min, val_min), (hi, 255, 255))
    return out


def close_and_dilate(mask: np.ndarray, kernel: int, iters: int, dilate_px: int) -> np.ndarray:
    if kernel > 1 and iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=iters)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask


def chroma_map(bgr: np.ndarray) -> np.ndarray:
    """Lab chroma per pixel (OpenCV 8-bit Lab scale, a/b offset by 128).

    Measured on the reference dataset, chroma is what actually separates the
    vibrant-orange gripper from warm backgrounds sharing its hue: gripper
    components sit at chroma 59-83 (49-88% of pixels >= 60), while the warm wall
    and wood measure 46-50 (0-23% >= 60) and corks 23-43 (0%). HSV saturation
    cannot make this cut — the wall is MORE HSV-saturated than the gripper in
    some frames.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.int16)
    a = lab[..., 1] - 128
    b = lab[..., 2] - 128
    return np.hypot(a, b)


def _grow_and_fill(keep: np.ndarray, grow: int) -> np.ndarray:
    """Dilate the kept region outward and fill every enclosed hole.

    Downstream gaussian-splat training corrupts on pinholes inside the gripper
    region, so the mask must be solid: grow closes boundary notches and swallows
    speck-sized misses; the flood fill turns any remaining enclosed white island
    black. Growing more than the gripper is acceptable; missing gripper is not.
    """
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        keep = cv2.dilate(keep, k)
    # Flood the background from the border; unreached zero-pixels are enclosed
    # holes inside kept regions.
    h, w = keep.shape
    flood = keep.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    for x, y in ((w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flood[y, x] == 0:
            cv2.floodFill(flood, ff_mask, (x, y), 255)
    holes = flood == 0
    keep = keep.copy()
    keep[holes] = 255
    return keep


def select_components(binary: np.ndarray, chroma: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Chroma-core gated selection; returns the kept region as a binary (255 = gripper).

    Every thresholded component above the min-area floor is judged by its core
    fraction — the share of its pixels that are vividly chromatic:

    - core fraction >= `core_frac`: the component is gripper; keep it whole
      (including its blur halo).
    - below the gate but containing gripper-sized vivid sub-regions: the component
      is a merge of gripper and background (a finger visually abutting the warm
      wall); cut out and keep the vivid sub-regions, drop the rest.
    - otherwise: background look-alike (cork, wood, wall); drop it.

    All passing components are kept — a motion-blurred finger can fragment into
    several blobs, and dropping any of them leaks gripper into training data.
    """
    h, w = binary.shape
    if cfg.roi is not None:
        x0, y0, x1, y1 = cfg.roi
        clipped = np.zeros_like(binary)
        clipped[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = binary[
            max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)
        ]
        binary = clipped

    core = ((chroma >= cfg.core_chroma) & (binary > 0)).astype(np.uint8) * 255
    min_area_px = cfg.min_area * h * w

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(binary)
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        comp = np.zeros_like(binary)
        cv2.drawContours(comp, [c], -1, 255, thickness=cv2.FILLED)
        comp_core = core & comp
        comp_px = int(np.count_nonzero(comp))
        frac = int(np.count_nonzero(comp_core)) / comp_px
        if frac >= cfg.core_frac:
            keep |= comp
        elif frac > 0:
            sub_contours, _ = cv2.findContours(
                comp_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for sub in sub_contours:
                if cv2.contourArea(sub) >= min_area_px:
                    cv2.drawContours(keep, [sub], -1, 255, thickness=cv2.FILLED)
    return keep


def process_image(bgr: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, bool]:
    """Run the full pipeline on one BGR image.

    Returns (mask, empty) — mask is single-channel at source resolution
    (0 = gripper, 255 = keep); empty is True when nothing was kept.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    base_hue, _, _ = rgb_to_hsv(cfg.color)
    binary = threshold(hsv, base_hue, cfg.hue_tol, cfg.sat_min, cfg.val_min)
    binary = close_and_dilate(binary, cfg.close_kernel, cfg.close_iters, cfg.dilate)
    keep = select_components(binary, chroma_map(bgr), cfg)
    empty = not np.any(keep)
    if not empty:
        keep = _grow_and_fill(keep, cfg.grow)
    mask = np.full(binary.shape, 255, dtype=np.uint8)
    mask[keep > 0] = 0
    return mask, empty
