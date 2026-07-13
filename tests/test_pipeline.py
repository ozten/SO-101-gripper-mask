"""Pipeline tests on synthetic images: solid-color shapes on a dark gray background."""

import cv2
import numpy as np
import pytest

from gripper_mask.pipeline import PipelineConfig, process_image, rgb_to_hsv

W, H = 320, 240  # min-area floor at defaults: 0.5% = 384 px


def hsv_bgr(h, s=255, v=255):
    """BGR tuple for an exact OpenCV HSV color."""
    px = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(px[0]), int(px[1]), int(px[2])


def rgb_of(bgr):
    return (bgr[2], bgr[1], bgr[0])


ORANGE_BGR = hsv_bgr(12)  # gripper stand-in, hue 12
ORANGE_RGB = rgb_of(ORANGE_BGR)


def canvas():
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    return img


def rect(img, x0, y0, x1, y1, bgr):
    img[y0:y1, x0:x1] = bgr


def cfg(**overrides):
    return PipelineConfig(color=ORANGE_RGB, **overrides)


def test_single_rectangle_masked_shape_and_polarity():
    img = canvas()
    rect(img, 100, 180, 180, H, ORANGE_BGR)  # touches bottom
    mask, empty = process_image(img, cfg())
    assert not empty
    assert mask.ndim == 2 and mask.dtype == np.uint8
    assert mask.shape == (H, W)
    assert mask[210, 140] == 0  # inside the blob
    assert mask[20, 20] == 255  # far corner untouched
    assert set(np.unique(mask)) <= {0, 255}


def test_two_blobs_second_at_forty_percent_kept():
    # Covers AE1: 2nd blob at ~40% of the largest's area -> both masked.
    img = canvas()
    rect(img, 40, 170, 110, H, ORANGE_BGR)  # 70x70
    rect(img, 220, 196, 264, H, ORANGE_BGR)  # 44x44 ~ 40%
    mask, _ = process_image(img, cfg())
    assert mask[210, 75] == 0
    assert mask[220, 242] == 0


def test_second_below_ratio_not_touching_bottom_excluded():
    img = canvas()
    rect(img, 40, 170, 140, H, ORANGE_BGR)  # 100x70 = 7000 px
    rect(img, 240, 60, 266, 86, ORANGE_BGR)  # 26x26 = 676 px < 15%, floats mid-frame
    mask, _ = process_image(img, cfg())
    assert mask[210, 90] == 0
    assert mask[73, 253] == 255


def test_second_below_ratio_touching_bottom_kept():
    img = canvas()
    rect(img, 40, 170, 140, H, ORANGE_BGR)
    rect(img, 240, 214, 266, H, ORANGE_BGR)  # small but real: touches bottom edge
    mask, _ = process_image(img, cfg())
    assert mask[210, 90] == 0
    assert mask[230, 253] == 0


def test_no_edge_prior_disables_bottom_fallback():
    img = canvas()
    rect(img, 40, 170, 140, H, ORANGE_BGR)
    rect(img, 240, 214, 266, H, ORANGE_BGR)
    mask, _ = process_image(img, cfg(edge_prior=False))
    assert mask[210, 90] == 0
    assert mask[230, 253] == 255  # fallback off: below-ratio blob dropped


def test_hue_wraparound_both_sides_detected():
    # Base near hue 3: the +/-8 window wraps below 0 -> [0..11] + [175..179].
    base = rgb_of(hsv_bgr(3))
    img = canvas()
    rect(img, 40, 170, 120, H, hsv_bgr(1))
    rect(img, 200, 170, 280, H, hsv_bgr(179))
    mask, _ = process_image(img, PipelineConfig(color=base))
    assert mask[210, 80] == 0
    assert mask[210, 240] == 0


def test_wood_panel_loses_to_core_score_despite_size_and_bottom_edge():
    # Mirrors frame 00000840: big warm panel ALSO touching the bottom edge.
    # Panel hue 19 passes the outer threshold but its value (170) sits below the
    # core-score value floor, so it scores ~0 while the gripper scores ~1.
    img = canvas()
    rect(img, 0, 60, 160, H, hsv_bgr(19, 180, 170))  # huge, bottom-touching panel
    rect(img, 230, 190, 290, H, ORANGE_BGR)  # small true gripper
    mask, _ = process_image(img, cfg())
    assert mask[220, 260] == 0  # gripper masked
    assert mask[150, 80] == 255  # panel NOT masked (neither primary nor rescued 2nd)


def test_cork_like_blob_excluded_by_saturation():
    # Corks share the gripper's hue but sit at saturation ~140 (< core floor 165).
    img = canvas()
    rect(img, 40, 120, 160, H, hsv_bgr(12, 140, 220))  # big bottom-touching "cork"
    rect(img, 230, 190, 290, H, ORANGE_BGR)  # smaller true gripper
    mask, _ = process_image(img, cfg())
    assert mask[220, 260] == 0  # gripper masked
    assert mask[200, 100] == 255  # cork-like blob excluded despite size + bottom edge


def test_interior_hole_filled():
    img = canvas()
    rect(img, 100, 150, 220, H, ORANGE_BGR)
    rect(img, 140, 180, 180, 210, (40, 40, 40))  # hole (screw/shadow)
    mask, _ = process_image(img, cfg(dilate=0, close_kernel=1, close_iters=0))
    assert mask[195, 160] == 0  # hole interior masked by filled contour


def test_below_min_area_gives_empty_all_white():
    img = canvas()
    rect(img, 100, 100, 115, 115, ORANGE_BGR)  # 225 px < 384 floor
    mask, empty = process_image(img, cfg(dilate=0))
    assert empty
    assert np.all(mask == 255)


def test_plain_frame_gives_empty_all_white():
    mask, empty = process_image(canvas(), cfg())
    assert empty
    assert np.all(mask == 255)


def test_roi_restricts_selection():
    img = canvas()
    rect(img, 20, 170, 100, H, ORANGE_BGR)  # outside ROI, bigger
    rect(img, 220, 190, 270, H, ORANGE_BGR)  # inside ROI
    mask, _ = process_image(img, cfg(roi=(160, 0, W, H)))
    assert mask[220, 245] == 0
    assert mask[210, 60] == 255


@pytest.mark.parametrize("hue,expected", [(12, [(4, 20)]), (3, [(0, 11), (175, 179)]), (176, [(0, 4), (168, 179)])])
def test_hue_windows_wrap_modulo_180(hue, expected):
    from gripper_mask.pipeline import hue_windows

    assert hue_windows(hue, 8) == expected


def test_rgb_to_hsv_orange():
    h, s, v = rgb_to_hsv(ORANGE_RGB)
    assert h == 12
    assert s == 255 and v == 255
