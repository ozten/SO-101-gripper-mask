"""Per-image mask pipeline: threshold, morphology, component selection, rendering.

Mask convention: single-channel uint8, 0 = gripper (masked area), 255 = keep.
Resize masks with nearest-neighbor only; bilinear creates gray boundary values.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Components whose lowest point reaches into this bottom band count as edge-touching.
BOTTOM_EDGE_FRACTION = 0.10
# A second component must be color-pure enough to be kept at all — via either the
# area-ratio or bottom-edge path. Without the gate, a large marginal background
# region (the wood panel is bigger than the gripper and touches the bottom edge)
# re-qualifies as component 2 through both paths. Relative floor guards blurry
# frames where every score is low; the absolute floor guards the warm-frame case
# where the panel scores ~0.15 while the gripper scores ~0.32.
SECOND_CORE_FLOOR = 0.5
SECOND_CORE_ABS = 0.25


@dataclass
class PipelineConfig:
    color: tuple[int, int, int]  # gripper base color, RGB
    hue_tol: int = 8  # OpenCV hue units (0-179 scale)
    sat_min: int = 100
    val_min: int = 60
    close_kernel: int = 5
    close_iters: int = 2
    dilate: int = 3
    second_ratio: float = 0.15
    min_area: float = 0.005  # fraction of frame pixels
    roi: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 (pixels)
    edge_prior: bool = True
    core_sat: int = 165  # core-score saturation floor (gripper ~200, corks ~140)
    core_val: int = 175  # core-score value floor (gripper bright plastic, wood ~145-165)


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


def _core_score(component_mask: np.ndarray, hsv: np.ndarray, hue: int, cfg: PipelineConfig) -> float:
    """Fraction of the component's pixels in the high-purity core window.

    Measured on the reference dataset, hue alone does NOT separate the gripper from
    warm backgrounds (corks and wood sit at the same hue); saturation separates the
    corks (~140 vs ~200) and value separates the wood (~145-165 vs bright plastic).
    The core window is the outer hue window with raised S/V floors; gripper
    components score 0.3-0.85, corks and wood score 0.0-0.17.
    """
    core = threshold(
        hsv,
        hue,
        cfg.hue_tol,
        max(cfg.sat_min, cfg.core_sat),
        max(cfg.val_min, cfg.core_val),
    )
    comp_px = int(np.count_nonzero(component_mask))
    if comp_px == 0:
        return 0.0
    return int(np.count_nonzero(core & component_mask)) / comp_px


def select_components(
    binary: np.ndarray, hsv: np.ndarray, base_hue: int, cfg: PipelineConfig
) -> list[np.ndarray]:
    """Pick the 1-2 contours to render, per KTD-4.

    ROI restriction -> min-area floor -> rank by banded core score (bottom-edge
    contact secondary, area tertiary) -> keep top; keep 2nd on area ratio or
    bottom-edge fallback, gated on color purity.
    """
    h, w = binary.shape
    if cfg.roi is not None:
        x0, y0, x1, y1 = cfg.roi
        clipped = np.zeros_like(binary)
        clipped[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = binary[
            max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)
        ]
        binary = clipped

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_px = cfg.min_area * h * w
    bottom_band = h * (1.0 - BOTTOM_EDGE_FRACTION)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        comp = np.zeros_like(binary)
        cv2.drawContours(comp, [c], -1, 255, thickness=cv2.FILLED)
        score = _core_score(comp, hsv, base_hue, cfg)
        touches_bottom = bool(c[:, :, 1].max() >= bottom_band)
        candidates.append((c, area, score, touches_bottom))

    if not candidates:
        return []

    edge_key = (lambda t: t[3]) if cfg.edge_prior else (lambda t: False)
    # Banded score (1 decimal) ranks by color purity without letting a 0.01 score
    # difference override a large area difference between two genuine fingers.
    candidates.sort(key=lambda t: (round(t[2], 1), edge_key(t), t[1]), reverse=True)

    kept = [candidates[0]]
    if len(candidates) > 1:
        primary = candidates[0]
        second = candidates[1]
        significant = second[1] >= cfg.second_ratio * primary[1]
        edge_rescue = cfg.edge_prior and second[3]
        pure_enough = second[2] >= max(SECOND_CORE_FLOOR * primary[2], SECOND_CORE_ABS)
        if (significant or edge_rescue) and pure_enough:
            kept.append(second)
    return [c for c, _, _, _ in kept]


def render_mask(shape: tuple[int, int], contours: list[np.ndarray]) -> np.ndarray:
    """White (255 = keep) canvas with kept components drawn as filled black (0) regions."""
    mask = np.full(shape, 255, dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 0, thickness=cv2.FILLED)
    return mask


def process_image(bgr: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, bool]:
    """Run the full pipeline on one BGR image.

    Returns (mask, empty) — mask is single-channel at source resolution;
    empty is True when no component survived (mask is all-white).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    base_hue, _, _ = rgb_to_hsv(cfg.color)
    binary = threshold(hsv, base_hue, cfg.hue_tol, cfg.sat_min, cfg.val_min)
    binary = close_and_dilate(binary, cfg.close_kernel, cfg.close_iters, cfg.dilate)
    kept = select_components(binary, hsv, base_hue, cfg)
    mask = render_mask(binary.shape, kept)
    return mask, not kept
