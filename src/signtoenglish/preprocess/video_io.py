"""Reading video files with real timestamps.

The reading API is two functions:

- `probe(path)` -> `VideoInfo` (size, nominal rate, duration; no decoding).
- `read_frames(path)` -> iterator of `Frame(image, t_ms)`, images as BGR uint8.

Rules:

- A frame's time is its presentation timestamp from the container, in whole milliseconds.
  It is never computed from a frame index and a frame rate, so variable-frame-rate files
  keep their real timing. The file's own start offset is kept (a file that starts at
  500 ms yields 500 first), which is also what `ffprobe` reports.
- Timestamps handed out are strictly increasing. A frame whose timestamp does not advance
  past the previous one (duplicates, or reordering in a damaged file) is dropped and
  counted in `ReadStats.dropped_non_increasing`.
- Rotation metadata is not applied; frames come out as stored.
- Pixels are converted to BGR the way OpenCV's reader does it, so a file read here and a
  file read with `cv2.VideoCapture` give the same arrays.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import av.error
import numpy as np
from av.video.reformatter import Interpolation

from signtoenglish.preprocess.schema import Frame

#: Colour conversion settings that make the BGR frames bit-identical to what
#: `cv2.VideoCapture` returns (measured on H.264 4:2:0 and 4:2:2 with OpenCV 5.0.0; the
#: parity test checks it). The reference pipeline decoded with OpenCV, and the pose
#: estimator's low-confidence keypoints are sensitive to one-level pixel differences.
_TO_BGR = Interpolation.POINT | Interpolation.ACCURATE_RND


class VideoError(RuntimeError):
    """The file is missing, is not a readable video, or has no video stream."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    #: Average frame rate declared by the container; a label only, never used for timing.
    nominal_fps: float | None
    #: Frame count declared by the container, if any. May be wrong; counting needs a decode.
    declared_frames: int | None
    duration_ms: int | None
    codec: str


@dataclass(slots=True)
class ReadStats:
    """Filled in while `read_frames` runs; complete once the iterator is exhausted."""

    decoded: int = 0
    yielded: int = 0
    dropped_non_increasing: int = 0
    dropped_without_timestamp: int = 0
    info: VideoInfo | None = field(default=None)


def _open(path: str | Path) -> av.container.InputContainer:
    p = Path(path)
    if not p.is_file():
        raise VideoError(f"no such video file: {p}")
    try:
        container = av.open(str(p), mode="r")
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise VideoError(f"cannot open {p} as video: {exc}") from exc
    if not container.streams.video:
        container.close()
        raise VideoError(f"{p} has no video stream")
    return container


def _info(container: av.container.InputContainer) -> VideoInfo:
    stream = container.streams.video[0]
    rate = stream.average_rate
    duration_ms: int | None = None
    if stream.duration is not None and stream.time_base is not None:
        duration_ms = round(float(stream.duration * stream.time_base) * 1000)
    elif container.duration is not None:
        duration_ms = round(container.duration / 1000)  # container.duration is microseconds
    return VideoInfo(
        width=int(stream.codec_context.width),
        height=int(stream.codec_context.height),
        nominal_fps=float(rate) if rate else None,
        declared_frames=int(stream.frames) if stream.frames else None,
        duration_ms=duration_ms,
        codec=str(stream.codec_context.name),
    )


def probe(path: str | Path) -> VideoInfo:
    """Container-level facts about the first video stream, without decoding."""
    with _open(path) as container:
        return _info(container)


def read_frames(path: str | Path, stats: ReadStats | None = None) -> Iterator[Frame]:
    """Decode the first video stream in presentation order.

    Yields `Frame(image, t_ms)` with `image` as `[H, W, 3]` uint8 BGR and `t_ms` the
    container timestamp in milliseconds, strictly increasing. Pass a `ReadStats` to learn
    how many frames were dropped. Raises `VideoError` if the file cannot be opened or a
    frame cannot be decoded.
    """
    stats = stats if stats is not None else ReadStats()
    with _open(path) as container:
        stats.info = _info(container)
        stream = container.streams.video[0]
        tb = stream.time_base
        time_base = Fraction(tb.numerator, tb.denominator) if tb is not None else None
        last_ms: int | None = None
        try:
            for frame in container.decode(stream):
                stats.decoded += 1
                if frame.pts is None or time_base is None:
                    stats.dropped_without_timestamp += 1
                    continue
                t_ms = round(frame.pts * time_base * 1000)  # exact rational arithmetic
                if last_ms is not None and t_ms <= last_ms:
                    stats.dropped_non_increasing += 1
                    continue
                last_ms = t_ms
                stats.yielded += 1
                image = np.ascontiguousarray(
                    frame.to_ndarray(format="bgr24", interpolation=_TO_BGR), dtype=np.uint8
                )
                yield Frame(image=image, t_ms=t_ms)
        except av.error.FFmpegError as exc:
            raise VideoError(f"decoding {path} failed after {stats.decoded} frames: {exc}") from exc
