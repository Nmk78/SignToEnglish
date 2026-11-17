"""Sentence-boundary detectors: the interface the streaming engine drives, and the two
simplest implementations.

Created in Step 3.1 (Track E) with the interface, `FixedLength`, and `OracleBoundaries`
only, because `infer/streaming.py` needs them. The file belongs to Track F (Step 3.2), which
adds the rest-position detector here and registers it in `build_detector`.

What a detector does, once per frame, in arrival order, causally:

    update(observation) -> BoundaryUpdate(signing, events)

- `events` are `start` and `end` marks with a time in the stream's own timestamps. A segment
  is the frames with `start <= t < end`, the same half-open rule as offline slicing.
- `signing` is the detector's claim that THIS frame shows signing. The engine adds up the
  time of the frames claimed as signing in each segment and never sends a segment with less
  than its configured minimum to the model, whatever `start` and `end` say. This is the
  guard against the model's habit of answering an empty scene with a fluent invented
  sentence (`docs/feasibility.md`, section 6.2). A detector cannot open a way around it; it
  can only make the claim well or badly.

The default claim, `hand_in_view`, is deliberately weak and is NOT a rest-position rule: a
hand counts when the pose estimator is confident about it and it lies inside the frame. It
is stricter than the base model's own 0.3 mask (a lowered hand can still score just above
0.3), but a hand resting in view passes it. Its two thresholds are [assumed] defaults, set
from the fixture measurements recorded in the Step 3.1 report; Step 3.2 chooses real ones
on validation takes and may override `frame_shows_signing` with stricter evidence.

This module may import only `common` (ARCHITECTURE.md, section 2), so it takes plain arrays
and repeats the two hand index ranges; a test checks them against `preprocess.schema`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field

from signtoenglish.common.config import FrozenConfig

#: COCO-WholeBody index ranges of the two hands (the same as `preprocess.schema.PART_INDICES`).
LEFT_HAND = (91, 112)
RIGHT_HAND = (112, 133)
NUM_KEYPOINTS = 133


class BoundaryError(ValueError):
    """A detector was configured or driven in a way it cannot handle."""


@dataclass(frozen=True, slots=True)
class Observation:
    """One frame as the detector sees it: raw pose-estimator output, nothing normalized."""

    t_ms: int
    keypoints: npt.NDArray[np.float32]  # [133, 2], in the units of `frame_size`
    scores: npt.NDArray[np.float32]  # [133]
    person_found: bool
    frame_size: tuple[int, int]  # (width, height)


@dataclass(frozen=True, slots=True)
class BoundaryEvent:
    kind: Literal["start", "end"]
    #: Stream time of the mark. It may lie before the newest frame, by at most the
    #: detector's `lookback_ms`; it never lies after it.
    t_ms: int


@dataclass(frozen=True, slots=True)
class BoundaryUpdate:
    signing: bool
    events: tuple[BoundaryEvent, ...] = ()


class SigningEvidenceConfig(FrozenConfig):
    """Thresholds of the default per-frame claim (`hand_in_view`). [assumed] defaults."""

    #: Mean pose-estimator score over a hand's 21 points must reach this.
    hand_score_min: float = Field(default=0.5, gt=0.0, le=1.0)
    #: At least this fraction of the hand's 21 points must lie inside the frame.
    hand_inside_min: float = Field(default=0.8, gt=0.0, le=1.0)


def hand_in_view(obs: Observation, cfg: SigningEvidenceConfig) -> bool:
    """True when at least one hand is confidently detected inside the frame."""
    width, height = obs.frame_size
    for first, last in (LEFT_HAND, RIGHT_HAND):
        if float(np.mean(obs.scores[first:last])) < cfg.hand_score_min:
            continue
        xy = obs.keypoints[first:last]
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        if float(np.mean(inside)) >= cfg.hand_inside_min:
            return True
    return False


class BoundaryDetector(ABC):
    """Base class of every detector. One instance serves one stream."""

    #: How far before the newest frame an event's time may lie. The engine keeps that much
    #: history while no segment is open. A detector with look-ahead (it decides about time
    #: `t` only after seeing frames up to `t + lookback_ms`) sets this.
    lookback_ms: int = 0

    def __init__(self, evidence: SigningEvidenceConfig | None = None) -> None:
        self.evidence = evidence or SigningEvidenceConfig()

    def frame_shows_signing(self, obs: Observation) -> bool:
        """The per-frame claim the engine's no-signing guard is built on."""
        return hand_in_view(obs, self.evidence)

    @abstractmethod
    def update(self, obs: Observation) -> BoundaryUpdate:
        """Take the next frame. Frames arrive with strictly increasing timestamps."""

    def notify_forced_cut(self, t_ms: int) -> None:  # noqa: B027 - optional hook
        """The engine cut the open segment at `t_ms` because it reached the maximum length,
        and opened the next one at `t_ms` itself. The detector's next `end` closes that one."""

    def notify_gap(self, last_t_ms: int, t_ms: int) -> None:  # noqa: B027 - optional hook
        """Frames between `last_t_ms` and `t_ms` never arrived. The engine closed any open
        segment at the last frame before the gap. Called before `update` for the frame at
        `t_ms`; a detector that considers a segment still under way must emit `start` again."""


# --- fixed length: the baseline ----------------------------------------------------------------


class FixedLengthConfig(FrozenConfig):
    kind: Literal["fixed_length"] = "fixed_length"
    #: Every segment is this long, back to back. It knows nothing about sentences.
    length_ms: int = Field(default=8000, gt=0)


class FixedLength(BoundaryDetector):
    """Cuts the stream into back-to-back windows of one length, whatever is in them. The
    baseline the rest-position detector has to beat, and the fallback until it exists."""

    def __init__(self, cfg: FixedLengthConfig, evidence: SigningEvidenceConfig | None = None):
        super().__init__(evidence)
        self.cfg = cfg
        self._start: int | None = None

    def update(self, obs: Observation) -> BoundaryUpdate:
        events: list[BoundaryEvent] = []
        if self._start is None:
            self._start = obs.t_ms
            events.append(BoundaryEvent("start", obs.t_ms))
        elif obs.t_ms - self._start >= self.cfg.length_ms:
            self._start = obs.t_ms
            events += [BoundaryEvent("end", obs.t_ms), BoundaryEvent("start", obs.t_ms)]
        return BoundaryUpdate(self.frame_shows_signing(obs), tuple(events))

    def notify_forced_cut(self, t_ms: int) -> None:
        self._start = t_ms

    def notify_gap(self, last_t_ms: int, t_ms: int) -> None:
        self._start = None


# --- oracle: boundaries given from outside ---------------------------------------------------


class OracleConfig(FrozenConfig):
    kind: Literal["oracle"] = "oracle"
    #: `[start_ms, end_ms)` windows in stream time, increasing and not overlapping.
    segments: tuple[tuple[int, int], ...]


class OracleBoundaries(BoundaryDetector):
    """Reports exactly the boundaries it was given (marked sentence times of a continuous
    take, or windows chosen for a test). For evaluation only: it separates what the engine
    and the model do from what a real detector gets wrong."""

    def __init__(self, cfg: OracleConfig, evidence: SigningEvidenceConfig | None = None):
        super().__init__(evidence)
        previous_end: int | None = None
        for start, end in cfg.segments:
            if end <= start:
                raise BoundaryError(f"oracle segment {start}..{end} ms is empty or reversed")
            if previous_end is not None and start < previous_end:
                raise BoundaryError(
                    f"oracle segments must increase without overlap; {start} ms starts before "
                    f"the previous one ends at {previous_end} ms"
                )
            previous_end = end
        self.cfg = cfg
        self._next = 0  # index of the first window that has not ended yet
        self._open = False
        self._resume = False  # a gap interrupted the open window

    def update(self, obs: Observation) -> BoundaryUpdate:
        events: list[BoundaryEvent] = []
        segments = self.cfg.segments
        while self._next < len(segments):
            start, end = segments[self._next]
            if obs.t_ms >= end:
                if self._open:
                    events.append(BoundaryEvent("end", end))
                    self._open = False
                self._next += 1
                continue
            if obs.t_ms >= start and not self._open:
                # After a gap the rest of the window starts at the first frame that arrived.
                events.append(BoundaryEvent("start", obs.t_ms if self._resume else start))
                self._open, self._resume = True, False
            break
        return BoundaryUpdate(self.frame_shows_signing(obs), tuple(events))

    def notify_gap(self, last_t_ms: int, t_ms: int) -> None:
        self._resume, self._open = self._open, False


# --- construction from config ------------------------------------------------------------------

#: Track F adds its detector's config class to this union and a branch to `build_detector`.
BoundaryConfig = Annotated[FixedLengthConfig | OracleConfig, Field(discriminator="kind")]


def build_detector(cfg: Any, evidence: SigningEvidenceConfig | None = None) -> BoundaryDetector:
    if isinstance(cfg, FixedLengthConfig):
        return FixedLength(cfg, evidence)
    if isinstance(cfg, OracleConfig):
        return OracleBoundaries(cfg, evidence)
    raise BoundaryError(f"no boundary detector for config {type(cfg).__name__}")


def parse_segments(text: str) -> tuple[tuple[int, int], ...]:
    """`"2000-9000,12000-20000"` -> `((2000, 9000), (12000, 20000))`, in milliseconds."""
    out: list[tuple[int, int]] = []
    for item in text.split(","):
        start, sep, end = item.strip().partition("-")
        try:
            if not sep:
                raise ValueError
            out.append((int(start), int(end)))
        except ValueError:
            raise BoundaryError(
                f"bad segment {item!r}: expected START-END in milliseconds, comma-separated"
            ) from None
    return tuple(out)


def segments_of(pairs: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(a), int(b)) for a, b in pairs)
