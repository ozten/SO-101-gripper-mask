"""Gallery tests: self-contained bundle, row states, regeneration, server errors."""

import socket

import cv2
import numpy as np
import pytest

from gripper_mask.cli import main

ORANGE_BGR = cv2.cvtColor(np.uint8([[[12, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
COLOR_ARG = "--color=#ff6600"


def write_frame(path, with_gripper=True):
    img = np.full((240, 320, 3), 40, dtype=np.uint8)
    if with_gripper:
        img[170:240, 100:180] = ORANGE_BGR
    cv2.imwrite(str(path), img)


@pytest.fixture
def dataset(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    write_frame(src / "0001.jpg")
    write_frame(src / "0002.jpg", with_gripper=False)  # will produce an empty mask
    main(["mask", str(src), str(out), COLOR_ARG])
    return src, out


def test_bundle_is_self_contained(dataset):
    src, out = dataset
    assert main(["qa", str(src), str(out), "--no-serve"]) == 0
    qa = out / "qa"
    html = (qa / "index.html").read_text()
    assert (qa / "originals" / "0001.jpg").exists()
    assert (qa / "masks" / "0001.png").exists()
    assert (qa / "overlays" / "0001.jpg").exists()
    # every reference is relative, inside qa/
    assert 'src="originals/0001.jpg"' in html
    assert 'src="masks/0001.png"' in html
    assert 'src="overlays/0001.jpg"' in html
    overlay = cv2.imread(str(qa / "overlays" / "0001.jpg"))
    original = cv2.imread(str(src / "0001.jpg"))
    assert overlay.shape == original.shape


def test_empty_mask_row_marked_and_prechecked_logic_present(dataset):
    src, out = dataset
    main(["qa", str(src), str(out), "--no-serve"])
    html = (out / "qa" / "index.html").read_text()
    assert 'id="row-0002"' in html
    assert 'class="row empty"' in html
    assert "EMPTY MASK" in html
    # pre-check default comes from the row's empty class when no stored flag exists
    assert "row.classList.contains('empty')" in html


def test_missing_mask_gets_placeholder_and_header_note(dataset, capsys):
    src, out = dataset
    write_frame(src / "0003.jpg")  # image added after mask run: no mask for it
    assert main(["qa", str(src), str(out), "--no-serve"]) == 0
    html = (out / "qa" / "index.html").read_text()
    assert "no mask generated" in html
    assert "1 image(s) have no mask" in html
    assert "1 missing masks" in capsys.readouterr().out


def test_stale_mask_gets_placeholder_row(dataset):
    src, out = dataset
    (src / "0001.jpg").unlink()  # mask 0001.png is now stale
    main(["qa", str(src), str(out), "--no-serve"])
    html = (out / "qa" / "index.html").read_text()
    assert "source image missing" in html
    assert "1 stale mask(s)" in html


def test_regeneration_refreshes_overlays(dataset):
    src, out = dataset
    main(["qa", str(src), str(out), "--no-serve"])
    before = (out / "qa" / "overlays" / "0001.jpg").read_bytes()
    # masks change after a re-tune: blank out the mask entirely
    mask = np.full((240, 320), 255, dtype=np.uint8)
    cv2.imwrite(str(out / "0001.png"), mask)
    main(["qa", str(src), str(out), "--no-serve"])
    after = (out / "qa" / "overlays" / "0001.jpg").read_bytes()
    assert before != after


def test_header_counts_and_disabled_copy_button(dataset):
    src, out = dataset
    main(["qa", str(src), str(out), "--no-serve"])
    html = (out / "qa" / "index.html").read_text()
    assert "const TOTAL = 2, EMPTY = 1;" in html
    assert '<button id="copy" disabled>' in html
    assert 'id="next-empty"' in html and 'id="next-flagged"' in html
    assert "navigator.clipboard" in html and "flaglist.select()" in html


def test_flag_keys_namespaced_by_dataset(dataset):
    src, out = dataset
    main(["qa", str(src), str(out), "--no-serve"])
    html = (out / "qa" / "index.html").read_text()
    assert 'const DATASET_ID = "' in html
    assert "DATASET_ID + ':' + row.dataset.file" in html


def test_port_in_use_exits_two(dataset, capsys):
    src, out = dataset
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    port = blocker.getsockname()[1]
    blocker.listen(1)
    try:
        code = main(["qa", str(src), str(out), "--port", str(port)])
    finally:
        blocker.close()
    assert code == 2
    assert "cannot bind" in capsys.readouterr().err
