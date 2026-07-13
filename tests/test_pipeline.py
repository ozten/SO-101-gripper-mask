"""Pipeline tests on synthetic images: solid-color shapes on a dark gray background."""

import cv2
import numpy as np
import pytest

from gripper_mask.pipeline import PipelineConfig, chroma_map, process_image, rgb_to_hsv

W, H = 320, 240  # min-area floor at defaults: 0.5% = 384 px


def hsv_bgr(h, s=255, v=255):
    """BGR tuple for an exact OpenCV HSV color."""
    px = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(px[0]), int(px[1]), int(px[2])


def rgb_of(bgr):
    return (bgr[2], bgr[1], bgr[0])


ORANGE_BGR = hsv_bgr(12)  # gripper stand-in, hue 12, vivid (Lab chroma well above 60)
ORANGE_RGB = rgb_of(ORANGE_BGR)
# Dull warm background stand-ins: same hue family, chroma below the core floor.
WALL_BGR = hsv_bgr(12, 150, 170)
CORK_BGR = hsv_bgr(12, 140, 200)


def test_standin_chromas_bracket_the_core_floor():
    # Guard the test fixtures themselves: the "gripper" color must be vivid and
    # the wall/cork stand-ins dull, mirroring the measured dataset separation.
    def chroma_of(bgr):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[:] = bgr
        return float(np.median(chroma_map(img)))

    assert chroma_of(ORANGE_BGR) >= 80
    assert chroma_of(WALL_BGR) < 55
    assert chroma_of(CORK_BGR) < 55


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
    rect(img, 100, 180, 180, H, ORANGE_BGR)
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
    rect(img, 40, 170, 110, H, ORANGE_BGR)
    rect(img, 220, 196, 264, H, ORANGE_BGR)
    mask, _ = process_image(img, cfg())
    assert mask[210, 75] == 0
    assert mask[220, 242] == 0


def test_all_vivid_fragments_kept():
    # A blurred finger can fragment into 3+ blobs; none may be dropped.
    img = canvas()
    rect(img, 30, 170, 100, H, ORANGE_BGR)
    rect(img, 140, 60, 180, 100, ORANGE_BGR)  # small, floating mid-frame
    rect(img, 230, 200, 280, H, ORANGE_BGR)
    mask, _ = process_image(img, cfg())
    assert mask[210, 60] == 0
    assert mask[80, 160] == 0
    assert mask[220, 255] == 0


def test_hue_wraparound_both_sides_detected():
    # Base near hue 3: the +/-8 window wraps below 0 -> [0..11] + [175..179].
    base = rgb_of(hsv_bgr(3))
    img = canvas()
    rect(img, 40, 170, 120, H, hsv_bgr(1))
    rect(img, 200, 170, 280, H, hsv_bgr(179))
    mask, _ = process_image(img, PipelineConfig(color=base))
    assert mask[210, 80] == 0
    assert mask[210, 240] == 0


def test_dull_wall_excluded_despite_size():
    # Mirrors frames 00000835/00000788: a huge warm wall shares the gripper's hue
    # but is dull (low Lab chroma) -> dropped by the core-fraction gate.
    img = canvas()
    rect(img, 0, 0, 180, H, WALL_BGR)  # wall fills the left half, touches bottom
    rect(img, 230, 190, 290, H, ORANGE_BGR)
    mask, _ = process_image(img, cfg())
    assert mask[220, 260] == 0  # gripper masked
    assert mask[120, 90] == 255  # wall untouched


def test_finger_merged_with_wall_is_cut_out():
    # Mirrors frame 00000319: a vivid finger visually abuts the dull wall, so
    # thresholding merges them into one component. The vivid sub-region must be
    # cut out and kept rather than dropped with the wall.
    img = canvas()
    rect(img, 0, 0, 165, H, WALL_BGR)
    rect(img, 155, 150, 215, H, ORANGE_BGR)  # finger overlapping the wall's edge
    mask, empty = process_image(img, cfg())
    assert not empty
    assert mask[200, 185] == 0  # finger masked despite the merge
    assert mask[120, 60] == 255  # wall body still excluded


def test_cork_like_blob_excluded():
    img = canvas()
    rect(img, 40, 120, 160, H, CORK_BGR)  # big bottom-touching "cork"
    rect(img, 230, 190, 290, H, ORANGE_BGR)
    mask, _ = process_image(img, cfg())
    assert mask[220, 260] == 0
    assert mask[200, 100] == 255


def test_interior_hole_filled():
    img = canvas()
    rect(img, 100, 120, 220, H, ORANGE_BGR)
    rect(img, 140, 160, 180, 200, (40, 40, 40))  # hole (screw/shadow)
    mask, _ = process_image(img, cfg(dilate=0, close_kernel=1, close_iters=0, grow=0))
    assert mask[180, 160] == 0  # hole interior masked even with grow disabled


def test_speck_holes_filled_and_mask_solid():
    # The gaussian-splat guarantee: scatter non-orange specks through the blob;
    # the final mask must be one solid region with zero enclosed white pixels.
    img = canvas()
    rect(img, 80, 100, 240, H, ORANGE_BGR)
    rng = np.random.default_rng(7)
    for _ in range(40):
        x = int(rng.integers(90, 230))
        y = int(rng.integers(110, 230))
        rect(img, x, y, x + 3, y + 3, (40, 40, 40))
    mask, _ = process_image(img, cfg())
    region = mask[110:230, 95:225]
    assert np.all(region == 0)  # no pinholes anywhere inside the blob


def test_grow_expands_mask_boundary():
    img = canvas()
    rect(img, 100, 100, 200, 180, ORANGE_BGR)
    tight, _ = process_image(img, cfg(grow=0, dilate=0, close_kernel=1, close_iters=0))
    grown, _ = process_image(img, cfg(grow=5, dilate=0, close_kernel=1, close_iters=0))
    assert tight[90, 150] == 255  # 10 px above the blob: outside tight mask
    assert grown[96, 150] == 0  # within 5 px of the blob edge: inside grown mask
    assert grown[80, 150] == 255  # well beyond the growth: still keep


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
    rect(img, 20, 170, 100, H, ORANGE_BGR)  # outside ROI
    rect(img, 220, 190, 270, H, ORANGE_BGR)  # inside ROI
    mask, _ = process_image(img, cfg(roi=(160, 0, W, H)))
    assert mask[220, 245] == 0
    assert mask[210, 60] == 255


@pytest.mark.parametrize(
    "hue,expected", [(12, [(4, 20)]), (3, [(0, 11), (175, 179)]), (176, [(0, 4), (168, 179)])]
)
def test_hue_windows_wrap_modulo_180(hue, expected):
    from gripper_mask.pipeline import hue_windows

    assert hue_windows(hue, 8) == expected


def test_rgb_to_hsv_orange():
    h, s, v = rgb_to_hsv(ORANGE_RGB)
    assert h == 12
    assert s == 255 and v == 255
