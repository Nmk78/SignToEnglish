"""The video writer stores real capture times and never leaves half-written files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signtoenglish.data.manifest import probe_video
from signtoenglish.record.writer import ClipWriter, WriterError


def _image(i: int, height: int = 240, width: int = 320) -> np.ndarray:
    image = np.full((height, width, 3), 50, dtype=np.uint8)
    image[20:60, 10 + i * 5 : 50 + i * 5] = 250
    return image


def test_container_carries_the_capture_times_exactly(tmp_path: Path, tools) -> None:
    # capture times in ns on an arbitrary monotonic clock, unevenly spaced, with a gap
    offsets_ms = [0, 31, 68, 99, 140, 171, 420, 455, 489, 2000, 2033]
    t0 = 5_000_000_123_456
    writer = ClipWriter(tmp_path / "s" / "clip.mp4")
    for i, offset in enumerate(offsets_ms):
        assert writer.write(_image(i), t0 + offset * 1_000_000) == offset
    path = writer.finish()

    assert path == tmp_path / "s" / "clip.mp4"
    assert tools.read_times_ms(path) == offsets_ms
    probe = probe_video(path)
    assert probe.n_frames == len(offsets_ms) and probe.strictly_increasing
    assert (probe.width, probe.height) == (320, 240)
    assert writer.n_frames == len(offsets_ms) and writer.n_dropped == 0
    assert writer.last_ms == 2033


def test_sub_millisecond_times_round_to_the_nearest_millisecond(tmp_path: Path, tools) -> None:
    writer = ClipWriter(tmp_path / "clip.mp4")
    for i, t_ns in enumerate([0, 33_400_000, 66_600_000, 100_499_999]):
        writer.write(_image(i), 1_000 + t_ns)
    assert tools.read_times_ms(writer.finish()) == [0, 33, 67, 100]


def test_frames_whose_time_does_not_advance_are_dropped_and_counted(tmp_path: Path, tools) -> None:
    writer = ClipWriter(tmp_path / "clip.mp4")
    results = [
        writer.write(_image(i), t_ms * 1_000_000) for i, t_ms in enumerate([10, 43, 43, 30, 77])
    ]
    assert results == [0, 33, None, None, 67]
    assert (writer.n_frames, writer.n_dropped) == (3, 2)
    assert tools.read_times_ms(writer.finish()) == [0, 33, 67]


def test_picture_content_survives_encoding(tmp_path: Path, tools) -> None:
    writer = ClipWriter(tmp_path / "clip.mp4")
    for i in range(6):
        writer.write(_image(i), i * 40_000_000)
    decoded = tools.read_images(writer.finish())
    assert len(decoded) == 6
    for i, image in enumerate(decoded):
        assert image.shape == (240, 320, 3)
        assert np.abs(image.astype(int) - _image(i).astype(int)).mean() < 3.0
    # frames are in order: frame 5 does not look like frame 0
    assert np.abs(decoded[5].astype(int) - _image(0).astype(int)).mean() > 3.0


def test_preparing_the_encoder_early_does_not_shift_the_times(tmp_path: Path, tools) -> None:
    writer = ClipWriter(tmp_path / "clip.mp4")
    writer.prepare(_image(0))  # as the recorder does when the countdown starts
    assert writer.n_frames == 0 and writer.last_ms == -1
    for i, t_ms in enumerate([3000, 3033, 3070]):  # the take starts three seconds later
        writer.write(_image(i), t_ms * 1_000_000)
    assert tools.read_times_ms(writer.finish()) == [0, 33, 70]

    unused = ClipWriter(tmp_path / "unused.mp4")
    unused.prepare(_image(0))
    unused.discard()  # a cancelled countdown
    assert [p.name for p in tmp_path.iterdir()] == ["clip.mp4"]


def test_odd_frame_sizes_lose_at_most_one_row_and_column(tmp_path: Path) -> None:
    writer = ClipWriter(tmp_path / "clip.mp4")
    for i in range(3):
        writer.write(_image(i, 241, 321), i * 40_000_000)
    probe = probe_video(writer.finish())
    assert (probe.width, probe.height) == (320, 240)


def test_nothing_is_left_behind_on_discard(tmp_path: Path) -> None:
    writer = ClipWriter(tmp_path / "clip.mp4")
    for i in range(120):  # enough for the encoder to have started writing the file
        writer.write(_image(i % 40), i * 40_000_000)
    assert writer.partial_path.is_file() and not writer.final_path.exists()
    writer.discard()
    writer.discard()
    assert list(tmp_path.iterdir()) == []


def test_partial_file_only_gets_its_final_name_when_promoted(tmp_path: Path, tools) -> None:
    writer = ClipWriter(tmp_path / "clip.mp4")
    for i in range(4):
        writer.write(_image(i), i * 40_000_000)
    partial = writer.close_partial()
    assert partial.name == "clip.partial.mp4" and partial.suffix == ".mp4"
    assert tools.read_times_ms(partial) == [0, 40, 80, 120]  # complete and playable for review
    assert not writer.final_path.exists()
    assert writer.promote() == writer.final_path
    assert [p.name for p in tmp_path.iterdir()] == ["clip.mp4"]


def test_refuses_to_overwrite_and_rejects_bad_input(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"existing")
    writer = ClipWriter(tmp_path / "clip.mp4")
    writer.write(_image(0), 0)
    writer.write(_image(1), 40_000_000)
    with pytest.raises(WriterError, match="overwrite"):
        writer.finish()
    assert (tmp_path / "clip.mp4").read_bytes() == b"existing"
    writer.discard()

    with pytest.raises(WriterError, match=r"\.mp4"):
        ClipWriter(tmp_path / "clip.mov")
    empty = ClipWriter(tmp_path / "empty.mp4")
    with pytest.raises(WriterError, match="no frames"):
        empty.finish()
    other = ClipWriter(tmp_path / "other.mp4")
    other.write(_image(0), 0)
    with pytest.raises(WriterError, match="size changed"):
        other.write(_image(1, 120, 160), 40_000_000)
    with pytest.raises(WriterError, match="BGR uint8"):
        other.write(np.zeros((240, 320), dtype=np.uint8), 80_000_000)
    other.discard()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]
