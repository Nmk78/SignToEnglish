"""Helpers for the streaming logic tests: synthetic keypoint frames, no files, no model.

These tests run in the quick check. The tests with the real model and real keypoints are in
tests/infer/test_streaming_model.py.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pytest

from signtoenglish.infer.streaming import DecodeJob, StreamConfig, StreamingEngine

FRAME_SIZE = (640, 480)


def _frame(hands: bool = True, score: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints in the middle of the frame; hand scores high (`hands`) or low."""
    keypoints = np.full((133, 2), (320.0, 240.0), dtype=np.float32)
    scores = np.full(133, 0.9, dtype=np.float32)
    scores[91:133] = score if score is not None else (0.9 if hands else 0.1)
    return keypoints, scores


def _times(seconds: float, fps: float = 30.0, start_ms: int = 0) -> list[int]:
    return [start_ms + round(i * 1000 / fps) for i in range(round(seconds * fps))]


def _frames(
    times: Iterable[int], hands: Callable[[int], bool] | bool = True
) -> list[tuple[int, np.ndarray, np.ndarray, bool]]:
    out = []
    for t in times:
        up = hands(t) if callable(hands) else hands
        keypoints, scores = _frame(up)
        out.append((t, keypoints, scores, True))
    return out


def _engine(max_frames: int = 256, detector: Any = None, **overrides: Any) -> StreamingEngine:
    cfg = StreamConfig.model_validate(overrides)
    return StreamingEngine(cfg, FRAME_SIZE, max_frames, detector)


def _label(job: DecodeJob) -> str:
    """Stands in for the translator in LOGIC tests only: text that names the job."""
    return f"{job.kind} {job.start_ms}-{job.end_ms} n={job.n_frames}"


@pytest.fixture
def make_frame() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    return _frame


@pytest.fixture
def times() -> Callable[..., list[int]]:
    return _times


@pytest.fixture
def frames() -> Callable[..., list[tuple[int, np.ndarray, np.ndarray, bool]]]:
    return _frames


@pytest.fixture
def engine() -> Callable[..., StreamingEngine]:
    return _engine


@pytest.fixture
def label() -> Callable[[DecodeJob], str]:
    return _label
