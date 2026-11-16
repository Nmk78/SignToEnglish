"""The webcam and the on-screen window for `s2e record`, on OpenCV.

Only this module touches the camera and the screen. It cannot be exercised by automated
tests on a machine without a camera and a person; the `Recorder` logic it plugs into is
tested with generated frames (tests/record/).

Timestamps: `CameraSource.read` stamps each frame with `time.monotonic_ns()` taken right
after the frame is grabbed from the driver. That is the frame's arrival time on this
machine's clock, not the sensor's exposure time; the difference is the camera pipeline's
delay, which is roughly constant within a take. No frame rate is assumed.

The picture in the window is mirrored so that it behaves like a mirror for the signer. The
saved video is not mirrored and has no text drawn on it.
"""

from __future__ import annotations

import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from signtoenglish.record.recorder import Frame, Overlay, RecorderError

WINDOW = "s2e record"
MIN_WIDTH, MIN_HEIGHT = 640, 480
PERMISSION_HINT = (
    "If macOS asked for camera access just now, allow it and run the command again. "
    "Otherwise open System Settings > Privacy & Security > Camera and enable the app you "
    "ran this from (Terminal, iTerm, or your editor), then restart that app."
)

_KEYS = {32: "space", 13: "enter", 10: "enter", 3: "enter", 27: "esc"}


class CameraSource:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        import cv2  # deferred: heavy

        self._cv2 = cv2
        self._cap: Any = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            raise RecorderError(f"cannot open camera {index}. {PERMISSION_HINT}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._failures = 0
        first = self._warm_up()
        self.height, self.width = first.shape[:2]
        if self.width < MIN_WIDTH or self.height < MIN_HEIGHT:
            self.close()
            raise RecorderError(
                f"camera {index} delivers {self.width}x{self.height}; "
                f"at least {MIN_WIDTH}x{MIN_HEIGHT} is needed"
            )

    def _warm_up(self) -> NDArray[np.uint8]:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            ok, image = self._cap.read()
            if ok and image is not None and image.size:
                return np.asarray(image, dtype=np.uint8)
            time.sleep(0.05)
        self.close()
        raise RecorderError(f"the camera opened but delivered no picture. {PERMISSION_HINT}")

    def read(self) -> Frame | None:
        while True:
            grabbed = self._cap.grab()
            t_ns = time.monotonic_ns()
            if grabbed:
                ok, image = self._cap.retrieve()
                if ok and image is not None and image.size:
                    self._failures = 0
                    return Frame(np.asarray(image, dtype=np.uint8), t_ns)
            self._failures += 1
            if self._failures >= 60:
                raise RecorderError("the camera stopped delivering frames")
            time.sleep(0.01)

    def measure_rate(self, n_frames: int = 30) -> float:
        """Frames per second actually delivered, measured over `n_frames` real frames."""
        times = []
        for _ in range(n_frames + 1):
            frame = self.read()
            if frame is not None:
                times.append(frame.t_ns)
        span = (times[-1] - times[0]) / 1e9
        return (len(times) - 1) / span if span > 0 else 0.0

    def close(self) -> None:
        self._cap.release()


def _ascii(text: str) -> str:
    # OpenCV's built-in fonts draw ASCII only.
    return text.encode("ascii", "replace").decode("ascii")


def _text(
    canvas: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, _ascii(text), (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(canvas, _ascii(text), (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _band(canvas: NDArray[np.uint8], top: int, bottom: int) -> None:
    top, bottom = max(0, top), min(canvas.shape[0], bottom)
    if bottom > top:
        canvas[top:bottom] = (canvas[top:bottom] * 0.35).astype(np.uint8)


def render_overlay(image: NDArray[np.uint8], overlay: Overlay) -> NDArray[np.uint8]:
    """The mirrored picture with the overlay drawn on a copy. `image` is not changed."""
    import cv2

    canvas: NDArray[np.uint8] = np.ascontiguousarray(image[:, ::-1]).copy()
    h, w = canvas.shape[:2]
    unit = h / 720.0
    margin = int(24 * unit)
    white, yellow, red, green = (255, 255, 255), (0, 220, 255), (40, 40, 230), (90, 220, 90)

    _band(canvas, 0, int(56 * unit))
    _text(canvas, overlay.heading, margin, int(38 * unit), 0.8 * unit, white)
    if overlay.state == "recording":
        cv2.circle(canvas, (w - int(190 * unit), int(28 * unit)), int(12 * unit), red, -1)
        elapsed = (overlay.elapsed_ms or 0) / 1000
        _text(canvas, f"REC {elapsed:4.1f}s", w - int(165 * unit), int(38 * unit), 0.8 * unit, red)
    elif overlay.state == "review":
        _text(canvas, "REVIEW", w - int(150 * unit), int(38 * unit), 0.8 * unit, green)

    chars = max(20, int(w / (22 * unit)))
    main_lines = textwrap.wrap(overlay.text, chars) or [""]
    detail_lines = [part for d in overlay.detail for part in textwrap.wrap(d, int(chars * 1.5))]
    tail_lines = ([overlay.notice] if overlay.notice else []) + (
        [overlay.keys] if overlay.keys else []
    )
    main_step, small_step = int(46 * unit), int(32 * unit)
    block = (
        len(main_lines) * main_step
        + (len(detail_lines) + len(tail_lines)) * small_step
        + int(28 * unit)
    )
    _band(canvas, h - block, h)
    y = h - block + int(44 * unit)
    for line in main_lines:
        _text(canvas, line, margin, y, 1.2 * unit, white, 3)
        y += main_step
    y -= int(8 * unit)
    for line in detail_lines:
        _text(canvas, line, margin, y, 0.75 * unit, (220, 220, 220))
        y += small_step
    if overlay.notice:
        _text(canvas, overlay.notice, margin, y, 0.75 * unit, yellow)
        y += small_step
    if overlay.keys:
        _text(canvas, overlay.keys, margin, y, 0.7 * unit, green)

    if overlay.big:
        scale = 9.0 * unit
        (tw, th), _ = cv2.getTextSize(overlay.big, cv2.FONT_HERSHEY_SIMPLEX, scale, 14)
        _text(canvas, overlay.big, (w - tw) // 2, (h + th) // 2, scale, yellow, 14)
    return canvas


class CvWindow:
    def __init__(self) -> None:
        import cv2

        self._cv2 = cv2
        self._shown = False
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def show(self, image: NDArray[np.uint8], overlay: Overlay) -> None:
        self._cv2.imshow(WINDOW, render_overlay(image, overlay))
        self._shown = True

    def poll_key(self) -> str | None:
        cv2 = self._cv2
        code = cv2.waitKey(1) & 0xFF
        if self._shown and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return "q"  # the window was closed
        if code == 0xFF:
            return None
        if code in _KEYS:
            return _KEYS[code]
        char = chr(code).lower()
        return char if char.isalnum() else None

    def replay(self, path: Path) -> None:
        """Play a recorded take at its real timing. Any key stops it."""
        import av

        cv2 = self._cv2
        overlay = Overlay("review", "Replay", "", keys="any key stops the replay")
        with av.open(str(path)) as container:
            started = time.monotonic()
            first_s: float | None = None
            for frame in container.decode(video=0):
                if frame.pts is None or frame.time_base is None:
                    continue
                at_s = float(frame.pts * frame.time_base)
                if first_s is None:
                    first_s = at_s
                wait_s = (at_s - first_s) - (time.monotonic() - started)
                image = np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)
                cv2.imshow(WINDOW, render_overlay(image, overlay))
                if cv2.waitKey(max(1, int(wait_s * 1000))) & 0xFF != 0xFF:
                    break

    def close(self) -> None:
        self._cv2.destroyAllWindows()
        self._cv2.waitKey(1)
