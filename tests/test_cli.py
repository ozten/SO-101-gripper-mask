"""CLI tests: directory flows, preflight guards, summary, exit codes."""

import cv2
import numpy as np
import pytest

from gripper_mask.cli import main

ORANGE_BGR = cv2.cvtColor(np.uint8([[[12, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
COLOR_ARG = "--color=#ff6600"  # hue ~12 at OpenCV scale, close enough to the test blobs


def write_frame(path, blob_x=100):
    img = np.full((240, 320, 3), 40, dtype=np.uint8)
    img[170:240, blob_x : blob_x + 80] = ORANGE_BGR
    cv2.imwrite(str(path), img)


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("bad", ["zzz", "#12345", "#12345g"])
def test_invalid_color_exits_two(tmp_path, bad):
    with pytest.raises(SystemExit) as exc:
        main(["mask", str(tmp_path), str(tmp_path / "out"), f"--color={bad}"])
    assert exc.value.code == 2


def test_happy_path_three_images(tmp_path, capsys):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for i in range(3):
        write_frame(src / f"{i:08d}.png")
    code = main(["mask", str(src), str(out), COLOR_ARG])
    assert code == 0
    masks = sorted(out.glob("*.png"))
    assert [m.name for m in masks] == ["00000000.png", "00000001.png", "00000002.png"]
    assert "processed 3, empty 0, skipped 0" in capsys.readouterr().out
    m = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
    assert m.shape == (240, 320)


def test_rerun_overwrites_by_default_and_skip_existing_skips(tmp_path):
    # Covers AE4: tuning iterations take effect without extra flags.
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    write_frame(src / "a.png", blob_x=40)
    assert main(["mask", str(src), str(out), COLOR_ARG]) == 0
    first = cv2.imread(str(out / "a.png"), cv2.IMREAD_GRAYSCALE)

    write_frame(src / "a.png", blob_x=200)  # scene changed
    assert main(["mask", str(src), str(out), COLOR_ARG]) == 0
    second = cv2.imread(str(out / "a.png"), cv2.IMREAD_GRAYSCALE)
    assert not np.array_equal(first, second)

    write_frame(src / "a.png", blob_x=40)
    assert main(["mask", str(src), str(out), COLOR_ARG, "--skip-existing"]) == 0
    third = cv2.imread(str(out / "a.png"), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(second, third)


def test_basename_collision_exits_two_writes_nothing(tmp_path, capsys):
    # Covers AE3.
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    write_frame(src / "a.jpg")
    write_frame(src / "a.png")
    code = main(["mask", str(src), str(out), COLOR_ARG])
    assert code == 2
    assert "a" in capsys.readouterr().err
    assert not out.exists()


def test_out_dir_equals_in_dir_exits_two(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    write_frame(src / "a.png")
    assert main(["mask", str(src), str(src), COLOR_ARG]) == 2


def test_missing_input_dir_exits_two(tmp_path):
    assert main(["mask", str(tmp_path / "nope"), str(tmp_path / "out"), COLOR_ARG]) == 2


def test_empty_input_dir_exits_two(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    assert main(["mask", str(src), str(tmp_path / "out"), COLOR_ARG]) == 2


def test_corrupt_file_warns_continues_exits_one(tmp_path, capsys):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    write_frame(src / "good.png")
    (src / "bad.jpg").write_bytes(b"this is not a jpeg")
    code = main(["mask", str(src), str(out), COLOR_ARG])
    assert code == 1
    captured = capsys.readouterr()
    assert "bad.jpg" in captured.err
    assert "processed 1, empty 0, skipped 1" in captured.out
    assert (out / "good.png").exists()
    assert not (out / "bad.png").exists()


def test_no_detection_writes_all_white_and_exits_one(tmp_path, capsys):
    # Covers AE2.
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    img = np.full((240, 320, 3), 40, dtype=np.uint8)  # no orange anywhere
    cv2.imwrite(str(src / "dark.png"), img)
    code = main(["mask", str(src), str(out), COLOR_ARG])
    assert code == 1
    captured = capsys.readouterr()
    assert "dark" in captured.err
    assert "empty 1" in captured.out
    m = cv2.imread(str(out / "dark.png"), cv2.IMREAD_GRAYSCALE)
    assert np.all(m == 255)


def test_non_image_files_ignored(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    write_frame(src / "a.png")
    (src / "notes.txt").write_text("hi")
    (src / "meta.json").write_text("{}")
    assert main(["mask", str(src), str(out), COLOR_ARG]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["a.png"]
