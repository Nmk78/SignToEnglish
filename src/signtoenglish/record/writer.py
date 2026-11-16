"""Video writer that stores each frame at its real capture time.

The file is H.264 in MP4 with a millisecond time base. Each frame's presentation time is
its capture timestamp relative to the first frame of the take, so the container carries a
variable frame rate exactly as the camera delivered it. No frame rate is assumed.

The file is written under a temporary name and only gets its final name in `finish()`;
`discard()` removes it. Both names end in `.mp4`, which git ignores everywhere.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

TIME_BASE = Fraction(1, 1000)
CODEC = "libx264"
# Near-lossless for keypoint extraction and fast enough to encode while capturing
# (measured 2026-11-16 on this Mac: 2.4 ms per 1280x720 frame on camera-like content).
CODEC_OPTIONS = {"preset": "veryfast", "crf": "18", "bf": "0"}


class WriterError(RuntimeError):
    pass


def temporary_path(final_path: Path) -> Path:
    return final_path.with_name(final_path.stem + ".partial.mp4")


class ClipWriter:
    """Encode BGR frames with capture timestamps in nanoseconds.

    `write` returns the frame's time in the file in milliseconds, or None when the frame
    was dropped because its timestamp did not advance by at least one millisecond.
    """

    def __init__(self, final_path: str | Path) -> None:
        self.final_path = Path(final_path)
        if self.final_path.suffix != ".mp4":
            raise WriterError(f"output must end in .mp4: {self.final_path}")
        self.partial_path = temporary_path(self.final_path)
        self._container: Any = None
        self._stream: Any = None
        self._t0_ns: int | None = None
        self._last_ms = -1
        self._size: tuple[int, int] | None = None  # width, height as encoded
        self.n_frames = 0
        self.n_dropped = 0
        self._closed = False

    @property
    def last_ms(self) -> int:
        """Time in the file of the last written frame; -1 before the first."""
        return self._last_ms

    @property
    def size(self) -> tuple[int, int] | None:
        return self._size

    def time_of(self, t_ns: int) -> int:
        """Capture time `t_ns` as milliseconds in the file (0 before the first frame)."""
        if self._t0_ns is None:
            return 0
        return max(0, round((t_ns - self._t0_ns) / 1_000_000))

    def _open(self, width: int, height: int) -> None:
        import av  # deferred: heavy

        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self.partial_path), "w", format="mp4")
        stream = self._container.add_stream(CODEC, options=dict(CODEC_OPTIONS))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = TIME_BASE
        stream.codec_context.time_base = TIME_BASE
        self._stream = stream
        self._size = (width, height)

    @staticmethod
    def _even_size(image: NDArray[np.uint8]) -> tuple[int, int]:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise WriterError(f"expected a BGR uint8 image, got {image.shape} {image.dtype}")
        # yuv420p needs even dimensions; drop at most one row and one column.
        height, width = image.shape[0] & ~1, image.shape[1] & ~1
        if width < 2 or height < 2:
            raise WriterError(f"frame is too small: {image.shape}")
        return width, height

    def prepare(self, image: NDArray[np.uint8]) -> None:
        """Open the file and the encoder for frames shaped like `image`, without writing it.

        Opening the encoder takes about 0.16 s (measured 2026-11-16). Doing it before the
        take starts keeps that pause out of the recording, where it would delay reading
        the camera and so distort the first frames' timestamps.
        """
        if self._closed:
            raise WriterError("writer is closed")
        if self._stream is None:
            self._open(*self._even_size(image))

    def write(self, image: NDArray[np.uint8], t_ns: int) -> int | None:
        import av

        if self._closed:
            raise WriterError("writer is closed")
        width, height = self._even_size(image)
        if self._stream is None:
            self._open(width, height)
        elif (width, height) != self._size:
            raise WriterError(f"frame size changed from {self._size} to {(width, height)}")
        if self._t0_ns is None:
            self._t0_ns = t_ns

        t_ms = self.time_of(t_ns)
        if t_ms <= self._last_ms:
            self.n_dropped += 1
            return None
        frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(image[:height, :width]), format="bgr24"
        )
        frame.pts = t_ms
        frame.time_base = TIME_BASE
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._last_ms = t_ms
        self.n_frames += 1
        return t_ms

    def _close_container(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._container is None:
            return
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()

    def finish(self) -> Path:
        """Flush, close, and move the file to its final name. Refuses to overwrite."""
        if self.n_frames == 0:
            self.discard()
            raise WriterError("no frames were written")
        self._close_container()
        if self.final_path.exists():
            raise WriterError(f"refusing to overwrite {self.final_path}")
        self.partial_path.rename(self.final_path)
        return self.final_path

    def close_partial(self) -> Path:
        """Flush and close but keep the temporary name, so the take can be reviewed."""
        if self.n_frames == 0:
            raise WriterError("no frames were written")
        self._close_container()
        return self.partial_path

    def promote(self) -> Path:
        """Give a closed temporary file its final name. Refuses to overwrite."""
        if not self._closed:
            raise WriterError("close the writer before promoting the file")
        if self.final_path.exists():
            raise WriterError(f"refusing to overwrite {self.final_path}")
        self.partial_path.rename(self.final_path)
        return self.final_path

    def discard(self) -> None:
        """Close and delete the temporary file. Safe to call more than once."""
        try:
            self._close_container()
        finally:
            self.partial_path.unlink(missing_ok=True)
