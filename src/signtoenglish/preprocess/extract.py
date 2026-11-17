"""Pose extraction: pixels in, raw whole-body keypoints out. The only copy.

One per-frame core, `PoseCore.process`, and one rate rule, `RateReducer`, are shared by
both entry points:

- `extract_file(path, core)` for recordings and uploaded clips;
- `LiveExtractor(core).push(image, t_ms)` for frames arriving from the camera.

Both thin frames to the model rate by timestamp BEFORE pose estimation and then call the
same core, so the same frames produce the same arrays either way.

The estimator is RTMPose whole-body through `rtmlib` on ONNX Runtime, because the base
model was trained on it. Following the reference pipeline (`docs/base_model_notes.md`,
sections 1 and 2.1): a person detector runs first; if it finds several people the largest
box is kept; if it finds no one the pose model runs on the whole frame. Output is raw: no
smoothing, no tracking, no flip, pixel coordinates of the frame as given.

Model files are local, pinned by SHA-256, and checked on load. Nothing is ever fetched.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from signtoenglish.common.hashing import verify_sha256
from signtoenglish.preprocess.schema import (
    NUM_KEYPOINTS,
    Frame,
    FrameKeypoints,
    ImageArray,
    KeypointConfig,
    PoseModelFile,
    RawKeypoints,
)
from signtoenglish.preprocess.video_io import ReadStats, read_frames

_RTMLIB_DEVICE = {"cpu": "cpu", "coreml": "mps"}


class RateReducer:
    """Thins a stream of timestamps to at most `model_rate_fps`, causally.

    A frame is kept when it arrives no earlier than `rate_slack` of a period before the
    next due time. Due times advance by one period per kept frame, and never fall more than
    the slack behind the frame just kept, so a source slightly slower than the model rate
    keeps every frame and a source twice as fast keeps every second one. The decision uses
    timestamps only, so the file path and the live path pick the same frames.
    """

    def __init__(self, model_rate_fps: float | None, rate_slack: float = 0.25) -> None:
        if model_rate_fps is not None and model_rate_fps <= 0:
            raise ValueError(f"model_rate_fps must be positive, got {model_rate_fps}")
        if not 0 <= rate_slack < 1:
            raise ValueError(f"rate_slack must be in [0, 1), got {rate_slack}")
        self._period_ms = None if model_rate_fps is None else 1000.0 / model_rate_fps
        self._slack = rate_slack
        self._next_due: float | None = None
        self._last_seen: int | None = None
        self.seen = 0
        self.kept = 0
        self.rejected_non_increasing = 0

    def keep(self, t_ms: int) -> bool:
        """Decide for the next frame. Timestamps must increase; one that does not is
        rejected and counted, which is how duplicates and reordering are handled live."""
        self.seen += 1
        if self._last_seen is not None and t_ms <= self._last_seen:
            self.rejected_non_increasing += 1
            return False
        self._last_seen = t_ms
        if self._period_ms is None:
            self.kept += 1
            return True
        slack_ms = self._slack * self._period_ms
        if self._next_due is not None and t_ms < self._next_due - slack_ms:
            return False
        due = t_ms if self._next_due is None else self._next_due
        self._next_due = max(due + self._period_ms, t_ms + self._period_ms - slack_ms)
        self.kept += 1
        return True


def _resolve(model: PoseModelFile, assets_root: Path) -> Path:
    path = assets_root / model.relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"pose model file not found: {path}. It is an approved local asset "
            "(OPEN_QUESTIONS.md, 'Approved assets'); this code never downloads it."
        )
    verify_sha256(path, model.sha256)
    return path


class PoseCore:
    """Detector plus whole-body pose model for single frames."""

    def __init__(self, cfg: KeypointConfig, assets_root: Path) -> None:
        from rtmlib import YOLOX, RTMPose  # heavy import, kept out of module import time

        self.cfg = cfg
        det_path = _resolve(cfg.detector, assets_root)
        pose_path = _resolve(cfg.pose, assets_root)
        with contextlib.redirect_stdout(io.StringIO()):  # rtmlib prints on load
            # The detector stays on CPU: ONNX Runtime's CoreML provider cannot run it.
            self._det: Any = YOLOX(
                str(det_path),
                model_input_size=cfg.detector.input_size,
                backend="onnxruntime",
                device="cpu",
            )
            self._pose: Any = RTMPose(
                str(pose_path),
                model_input_size=cfg.pose.input_size,
                to_openpose=False,
                backend="onnxruntime",
                device=_RTMLIB_DEVICE[cfg.pose_provider],
            )
        self.providers: dict[str, list[str]] = {
            "detector": list(self._det.session.get_providers()),
            "pose": list(self._pose.session.get_providers()),
        }

    def process(self, image: ImageArray, t_ms: int) -> FrameKeypoints:
        """Raw keypoints for one BGR frame. Never raises on an empty scene."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"expected a [H, W, 3] uint8 BGR image, got {image.shape} {image.dtype}"
            )
        boxes = np.asarray(self._det(image))
        person_found = boxes.ndim == 2 and boxes.shape[0] > 0
        if person_found:
            # The base model's rule: keep the largest box (bundled yolox.py:135-138).
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(np.argmax(areas))
            chosen: Any = boxes[best : best + 1]
        else:
            chosen = []  # the pose model then takes the whole frame (rtmpose.py:25-26)
        keypoints, scores = self._pose(image, bboxes=chosen)
        if keypoints.shape != (1, NUM_KEYPOINTS, 2) or scores.shape != (1, NUM_KEYPOINTS):
            raise RuntimeError(f"unexpected pose output {keypoints.shape}, {scores.shape}")
        return FrameKeypoints(
            t_ms=t_ms,
            keypoints=keypoints[0].astype(np.float32),
            scores=scores[0].astype(np.float32),
            person_found=bool(person_found),
        )


def iter_keypoints(
    frames: Iterable[Frame], core: PoseCore, reducer: RateReducer | None = None
) -> Iterator[FrameKeypoints]:
    """Rate reduction, then the core, over any source of frames."""
    reducer = reducer or RateReducer(core.cfg.model_rate_fps, core.cfg.rate_slack)
    for frame in frames:
        if reducer.keep(frame.t_ms):
            yield core.process(frame.image, frame.t_ms)


def extract_file(path: str | Path, core: PoseCore, stats: ReadStats | None = None) -> RawKeypoints:
    """Raw keypoints for every kept frame of a video file, with container timestamps."""
    stats = stats if stats is not None else ReadStats()
    out = list(iter_keypoints(read_frames(path, stats), core))
    assert stats.info is not None
    return RawKeypoints.from_frames(out, stats.info.width, stats.info.height)


class LiveExtractor:
    """The live entry point: push frames as they arrive, with their capture timestamps."""

    def __init__(self, core: PoseCore) -> None:
        self._core = core
        self.reducer = RateReducer(core.cfg.model_rate_fps, core.cfg.rate_slack)

    def push(self, image: ImageArray, t_ms: int) -> FrameKeypoints | None:
        """Keypoints for this frame, or None if the rate rule or a non-increasing
        timestamp drops it."""
        if not self.reducer.keep(t_ms):
            return None
        return self._core.process(image, t_ms)
