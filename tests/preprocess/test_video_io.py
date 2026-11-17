"""`video_io`: real timestamps from the container, on synthetic files written with known
times and on the fixture clips."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from signtoenglish.preprocess.video_io import ReadStats, VideoError, probe, read_frames


def _frames(n: int, width: int = 160, height: int = 120) -> list[np.ndarray]:
    out = []
    for i in range(n):
        image = np.zeros((height, width, 3), np.uint8)
        image[:, : 10 + 5 * i] = (40, 90, 200)
        out.append(image)
    return out


def test_constant_rate_timestamps_are_read_back(
    tmp_path: Path, video_writer: Callable[..., Path]
) -> None:
    written = [round(i * 1000 / 30) for i in range(20)]
    path = video_writer(tmp_path / "cfr.mkv", _frames(20), written)
    stats = ReadStats()
    frames = list(read_frames(path, stats))
    assert [f.t_ms for f in frames] == written
    assert stats.decoded == stats.yielded == 20 and stats.dropped_non_increasing == 0
    assert frames[0].image.shape == (120, 160, 3) and frames[0].image.dtype == np.uint8
    info = probe(path)
    assert (info.width, info.height) == (160, 120)


def test_variable_rate_keeps_real_timestamps(
    tmp_path: Path, video_writer: Callable[..., Path], ffprobe_ms: Callable[[Path], list[float]]
) -> None:
    written = [0, 33, 100, 117, 250, 283, 400, 900, 933, 1000]
    path = video_writer(tmp_path / "vfr.mkv", _frames(len(written)), written)
    got = [f.t_ms for f in read_frames(path)]
    assert got == written  # not 0, 33, 67, ...: nothing is derived from a frame rate
    assert len(set(np.diff(got))) > 3
    reference = ffprobe_ms(path)
    assert len(reference) == len(got)
    assert max(abs(a - b) for a, b in zip(got, reference, strict=True)) <= 1.0


def test_start_offset_is_kept(tmp_path: Path, video_writer: Callable[..., Path]) -> None:
    written = [500, 540, 580, 620]
    path = video_writer(tmp_path / "offset.mkv", _frames(4), written)
    assert [f.t_ms for f in read_frames(path)] == written


def test_images_are_bgr(tmp_path: Path, video_writer: Callable[..., Path]) -> None:
    red = np.zeros((120, 160, 3), np.uint8)
    red[..., 2] = 230  # BGR: channel 2 is red
    path = video_writer(tmp_path / "red.mkv", [red] * 3, [0, 33, 67])
    image = next(iter(read_frames(path))).image
    b, g, r = (float(image[..., c].mean()) for c in range(3))
    assert r > 180 and b < 60 and g < 60


def test_missing_and_unreadable_files_raise_video_error(tmp_path: Path) -> None:
    with pytest.raises(VideoError, match="no such video file"):
        list(read_frames(tmp_path / "nope.mp4"))
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 50)
    with pytest.raises(VideoError, match="cannot open"):
        probe(junk)


@pytest.mark.parametrize("stem", ["hands_visible", "hands_leave"])
def test_fixture_clip_timestamps_match_ffprobe(
    stem: str, clip_path: Callable[[str], Path], ffprobe_ms: Callable[[Path], list[float]]
) -> None:
    path = clip_path(stem)
    stats = ReadStats()
    got = np.array([f.t_ms for f in read_frames(path, stats)])
    reference = np.array(ffprobe_ms(path))
    assert stats.dropped_non_increasing == 0 and stats.dropped_without_timestamp == 0
    assert len(got) == len(reference) == 150
    assert np.all(np.diff(got) > 0)
    assert float(np.abs(got - reference).max()) <= 1.0


@pytest.mark.parametrize("stem", ["hands_visible", "hands_leave"])
def test_frames_equal_opencv_decoding(stem: str, clip_path: Callable[[str], Path]) -> None:
    """The reference pipeline decoded with `cv2.VideoCapture`. Our reader must hand the pose
    estimator the same pixels: on low-confidence keypoints a one-level difference is enough
    to move a point across the frame."""
    import cv2

    path = clip_path(stem)
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    n = 0
    for frame in read_frames(path):
        ok, expected = cap.read()
        assert ok
        assert np.array_equal(frame.image, expected), f"frame {n} differs from OpenCV's"
        n += 1
    assert not cap.read()[0] and n == 150
    cap.release()
