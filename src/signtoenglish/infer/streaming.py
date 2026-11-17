"""Streaming engine: keypoint frames in, partial and committed text out (ARCHITECTURE.md 6.1).

    push(t_ms, keypoints, scores)
      -> rolling buffer -> boundary detector
      every HOP while a segment is open:  request a PARTIAL decode of segment_start..now
      at a segment end (or the maximum):  queue a FINAL decode of segment_start..end

The engine is a state machine with no model, no clock, no thread, and no transport in it.
It never translates anything itself: it hands out `DecodeJob`s (`next_job`) and takes the
text back (`complete`). A replay loop (`replay`, below) or the service's single decode
worker sits in between. That is what makes "the decode took longer than the hop" a plain
state: while a job is out, a hop that comes due is skipped and counted.

Rules the engine enforces, each with a counter in `StreamStats`:

- **Real timestamps.** Every decision uses the frames' own timestamps. No frame rate is
  assumed anywhere. A frame whose timestamp repeats the previous one (duplicate) or lies
  before it (out of order) is dropped. A jump of more than `max_gap_ms` is a gap: the open
  segment is closed at the last frame before it, because the signing in between is lost.
- **Committed text never changes.** A segment gets one id and at most one FINAL. A partial
  result that comes back after its segment has closed is discarded, so no PARTIAL can
  follow that segment's FINAL.
- **Nothing without signing reaches the model.** A job, partial or final, is only created
  when the segment is at least `min_segment_ms` long AND holds at least `min_signing_ms`
  of frames the boundary detector claims as signing (`models/boundary.py`). The base model
  answers an empty scene with a fluent invented sentence, so this guard is not optional
  and a detector cannot switch it off.
- **The base model's maximum.** A segment is cut when it reaches `max_segment_ms` or the
  model's maximum frame count, whichever comes first, and the next one opens at the cut.
  A streamed segment is therefore never thinned.
- **Bounded memory.** The buffer holds the open segment (at most the model's maximum plus
  one frame) or, between segments, the detector's look-back. Queued finals are capped at
  `max_pending_finals`; beyond that the oldest is dropped, reported, and counted.

A segment is the frames with `start_ms <= t < end_ms`, the same half-open slice as
`OfflineTranslator.translate_raw`, and decoding goes through
`OfflineTranslator.translate_arrays`. With the same boundaries, streaming and offline
therefore give the model identical arrays.

Thread use: `push`, `flush`, `next_job`, `complete`, and `fail` take one lock, so a
receiving loop and one decode worker may share an engine. Only one job is out at a time.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field

from signtoenglish.common.config import FrozenConfig, PathsConfig, find_repo_root, load_config
from signtoenglish.infer.offline import OfflineTranslator
from signtoenglish.models.boundary import (
    BoundaryConfig,
    BoundaryDetector,
    FixedLengthConfig,
    Observation,
    SigningEvidenceConfig,
    build_detector,
)
from signtoenglish.models.translator import DecodeConfig
from signtoenglish.preprocess.cache import KeypointCache, Source, get_or_extract
from signtoenglish.preprocess.schema import NUM_KEYPOINTS, KeypointConfig
from signtoenglish.preprocess.schema import RawKeypoints as RawKeypoints  # re-exported for `eval`

DEFAULT_STREAM_CONFIG = Path("configs") / "stream" / "default.yaml"
#: How many committed sentences the engine itself remembers (`StreamingEngine.committed`).
#: Whoever needs the whole transcript keeps the events it was handed.
COMMITTED_HISTORY = 256

EventType = Literal["partial", "final", "dropped", "error"]
#: Why a segment was closed.
ClosedBy = Literal["boundary", "max_length", "gap", "flush"]
#: Why a closed segment produced no text.
DropReason = Literal["empty", "too_short", "no_signing", "backlog"]
Pace = Literal["instant", "simulated", "realtime"]


class StreamError(RuntimeError):
    """The engine was driven in a way it cannot handle; the message says how."""


class StreamConfig(FrozenConfig):
    """`configs/stream/default.yaml`. All times are milliseconds of stream time.
    Every default is [assumed] until chosen on validation takes (Steps 3.2 and 3.3)."""

    #: A partial decode of the open segment is requested this often.
    hop_ms: int = Field(default=1000, gt=0)
    #: A segment is cut at this length, or at the model's maximum frame count if sooner.
    max_segment_ms: int = Field(default=16000, gt=0)
    #: A shorter segment is dropped without decoding.
    min_segment_ms: int = Field(default=1000, ge=0)
    #: A segment with less signing than this is dropped without decoding.
    min_signing_ms: int = Field(default=500, gt=0)
    #: A larger jump between consecutive timestamps is a gap.
    max_gap_ms: int = Field(default=500, gt=0)
    #: More finals than this waiting for the decoder: the oldest is dropped and reported.
    max_pending_finals: int = Field(default=8, ge=1)
    evidence: SigningEvidenceConfig = SigningEvidenceConfig()
    boundary: BoundaryConfig = FixedLengthConfig()


def load_stream_config(path: str | Path | None = None) -> StreamConfig:
    return load_config(
        Path(path) if path else find_repo_root() / DEFAULT_STREAM_CONFIG, StreamConfig
    )


@dataclass(frozen=True, slots=True)
class DecodeJob:
    """One request to translate a run of frames. The arrays are copies; the engine's
    buffer may move on while the job is out."""

    kind: Literal["partial", "final"]
    segment_id: int
    start_ms: int
    #: Exclusive. For a partial: one past the newest frame included.
    end_ms: int
    timestamps_ms: npt.NDArray[np.int64]  # [T]
    keypoints: npt.NDArray[np.float32]  # [T, 133, 2]
    scores: npt.NDArray[np.float32]  # [T, 133]
    frame_size: tuple[int, int]
    signing_ms: int
    closed_by: ClosedBy | None  # None for a partial

    @property
    def n_frames(self) -> int:
        return int(self.timestamps_ms.shape[0])


@dataclass(frozen=True, slots=True)
class StreamEvent:
    type: EventType
    segment_id: int
    text: str
    start_ms: int
    end_ms: int
    n_frames: int
    signing_ms: int
    closed_by: ClosedBy | None = None
    reason: str | None = None  # for "dropped" (a `DropReason`) and "error" (the detail)
    decode_ms: float | None = None
    #: Stream time from `end_ms` to the moment the text was available, when the caller
    #: told `complete` what time it was. For a final this is the commit delay.
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(slots=True)
class StreamStats:
    frames_pushed: int = 0
    frames_accepted: int = 0
    duplicate_timestamps: int = 0
    out_of_order_timestamps: int = 0
    gaps: int = 0
    segments_opened: int = 0
    segments_closed_by_boundary: int = 0
    segments_closed_by_max_length: int = 0
    segments_closed_by_gap: int = 0
    segments_closed_by_flush: int = 0
    segments_dropped_empty: int = 0
    segments_dropped_too_short: int = 0
    segments_dropped_no_signing: int = 0
    finals_dropped_backlog: int = 0
    finals_queued: int = 0
    finals_emitted: int = 0
    hops_due: int = 0
    hops_skipped_busy: int = 0
    hops_skipped_no_signing: int = 0
    partials_requested: int = 0
    partials_replaced_before_decoding: int = 0
    partials_emitted: int = 0
    partials_discarded_stale: int = 0
    decode_failures: int = 0
    boundary_events_ignored: int = 0
    max_buffered_frames: int = 0
    max_pending_finals: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class _Frame:
    t_ms: int
    keypoints: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]
    #: Stream time this frame adds to its segment's signing total: the time since the
    #: previous frame if the detector claims this frame as signing, otherwise 0.
    signing_credit_ms: int


class StreamingEngine:
    """One per session. See the module docstring for the rules."""

    def __init__(
        self,
        cfg: StreamConfig,
        frame_size: tuple[int, int],
        max_frames: int,
        detector: BoundaryDetector | None = None,
    ) -> None:
        if max_frames < 1:
            raise StreamError(f"max_frames must be positive, got {max_frames}")
        if frame_size[0] <= 0 or frame_size[1] <= 0:
            raise StreamError(f"bad frame size {frame_size}")
        self.cfg = cfg
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.max_frames = max_frames
        self.detector = detector or build_detector(cfg.boundary, cfg.evidence)
        self.stats = StreamStats()
        self._lock = threading.RLock()
        self._buffer: deque[_Frame] = deque()
        self._last_t: int | None = None
        self._signing = False
        self._next_id = 0
        self._seg_id: int | None = None  # the open segment
        self._seg_start = 0
        self._seg_frames = 0
        self._next_hop = 0
        self._last_end: int | None = None
        self._finals: deque[DecodeJob] = deque()
        self._partial: DecodeJob | None = None
        self._in_flight: DecodeJob | None = None
        self._committed: deque[StreamEvent] = deque(maxlen=COMMITTED_HISTORY)

    # --- state the caller may read -----------------------------------------------------------

    @property
    def signing(self) -> bool:
        """The detector's claim about the newest frame (for a status line)."""
        return self._signing

    @property
    def segment_open(self) -> bool:
        return self._seg_id is not None

    @property
    def buffered_frames(self) -> int:
        return len(self._buffer)

    @property
    def pending_finals(self) -> int:
        return len(self._finals)

    @property
    def busy(self) -> bool:
        """A job is out, or finals are waiting. A hop that comes due now is skipped."""
        return self._in_flight is not None or bool(self._finals)

    @property
    def committed(self) -> tuple[StreamEvent, ...]:
        """The most recent FINALs (up to `COMMITTED_HISTORY`), in commit order. An entry is
        never altered or replaced; the oldest fall off the front."""
        return tuple(self._committed)

    # --- frames in ---------------------------------------------------------------------------

    def push(
        self,
        t_ms: int,
        keypoints: npt.NDArray[np.floating],
        scores: npt.NDArray[np.floating],
        person_found: bool = True,
    ) -> list[StreamEvent]:
        """Take one frame of raw keypoints with its capture or container timestamp.
        Returns the events this frame caused directly (segments dropped without decoding);
        text arrives through `complete`."""
        if keypoints.shape != (NUM_KEYPOINTS, 2) or scores.shape != (NUM_KEYPOINTS,):
            raise StreamError(
                f"expected keypoints [{NUM_KEYPOINTS}, 2] and scores [{NUM_KEYPOINTS}], "
                f"got {keypoints.shape} and {scores.shape}"
            )
        t = int(t_ms)
        with self._lock:
            stats = self.stats
            stats.frames_pushed += 1
            events: list[StreamEvent] = []
            last = self._last_t
            if last is not None:
                if t == last:
                    stats.duplicate_timestamps += 1
                    return events
                if t < last:
                    stats.out_of_order_timestamps += 1
                    return events
            gap = last is not None and t - last > self.cfg.max_gap_ms
            if gap:
                assert last is not None
                stats.gaps += 1
                if self._seg_id is not None:
                    self._close(last + 1, "gap", events)
                self.detector.notify_gap(last, t)
            stats.frames_accepted += 1
            self._last_t = t

            kp = np.array(keypoints, dtype=np.float32)  # copies: the caller may reuse its arrays
            sc = np.array(scores, dtype=np.float32)
            update = self.detector.update(
                Observation(t, kp, sc, bool(person_found), self.frame_size)
            )
            self._signing = bool(update.signing)
            credit = t - last if (update.signing and last is not None and not gap) else 0
            self._buffer.append(_Frame(t, kp, sc, credit))
            if self._seg_id is not None:
                self._seg_frames += 1

            for event in update.events:
                if event.t_ms > t:
                    raise StreamError(
                        f"boundary detector reported a {event.kind} at {event.t_ms} ms, "
                        f"after the newest frame at {t} ms"
                    )
                if event.kind == "start":
                    if self._seg_id is not None:
                        stats.boundary_events_ignored += 1
                    else:
                        self._open(event.t_ms)
                elif self._seg_id is None:
                    stats.boundary_events_ignored += 1
                else:
                    self._close(max(event.t_ms, self._seg_start), "boundary", events)

            if self._seg_id is not None and (
                self._seg_frames > self.max_frames or t - self._seg_start >= self.cfg.max_segment_ms
            ):
                # The newest frame is the first of the next segment.
                self._close(t, "max_length", events)
                self._open(t)
                self.detector.notify_forced_cut(t)

            if self._seg_id is not None and t >= self._next_hop:
                self._hop(t)
            self._trim(t)
            stats.max_buffered_frames = max(stats.max_buffered_frames, len(self._buffer))
            return events

    def flush(self) -> list[StreamEvent]:
        """End of input (a `stop` message, or the end of a file): close the open segment
        at the newest frame. Its text, if any, still arrives through `complete`."""
        with self._lock:
            events: list[StreamEvent] = []
            if self._seg_id is not None and self._last_t is not None:
                self._close(self._last_t + 1, "flush", events)
            return events

    # --- decode jobs out, text in --------------------------------------------------------

    def next_job(self) -> DecodeJob | None:
        """The next thing to translate, finals before partials; None when there is nothing
        or a job is already out."""
        with self._lock:
            if self._in_flight is not None:
                return None
            if self._finals:
                self._in_flight = self._finals.popleft()
            elif self._partial is not None:
                self._in_flight, self._partial = self._partial, None
            return self._in_flight

    def complete(
        self, job: DecodeJob, text: str, decode_ms: float, now_ms: int | None = None
    ) -> list[StreamEvent]:
        """Hand back the text of the job that is out. `now_ms` is the stream time at which
        the text became available, if the caller knows it; it sets `latency_ms`."""
        with self._lock:
            self._take_back(job)
            latency = None if now_ms is None else float(max(0, now_ms - job.end_ms))
            if job.kind == "partial" and job.segment_id != self._seg_id:
                self.stats.partials_discarded_stale += 1
                return []
            event = StreamEvent(
                type=job.kind,
                segment_id=job.segment_id,
                text=text,
                start_ms=job.start_ms,
                end_ms=job.end_ms,
                n_frames=job.n_frames,
                signing_ms=job.signing_ms,
                closed_by=job.closed_by,
                decode_ms=float(decode_ms),
                latency_ms=latency,
            )
            if job.kind == "final":
                self.stats.finals_emitted += 1
                self._committed.append(event)
            else:
                self.stats.partials_emitted += 1
            return [event]

    def fail(self, job: DecodeJob, detail: str) -> list[StreamEvent]:
        """The job that is out could not be translated. Nothing is committed for it."""
        with self._lock:
            self._take_back(job)
            self.stats.decode_failures += 1
            return [
                StreamEvent(
                    type="error",
                    segment_id=job.segment_id,
                    text="",
                    start_ms=job.start_ms,
                    end_ms=job.end_ms,
                    n_frames=job.n_frames,
                    signing_ms=job.signing_ms,
                    closed_by=job.closed_by,
                    reason=f"{job.kind} decode failed: {detail}",
                )
            ]

    # --- internals (called with the lock held) ---------------------------------------------

    def _take_back(self, job: DecodeJob) -> None:
        if self._in_flight is not job:
            raise StreamError("this job is not the one that is out")
        self._in_flight = None

    def _open(self, start_ms: int) -> None:
        if self._last_end is not None:
            start_ms = max(start_ms, self._last_end)
        self._seg_id = self._next_id
        self._next_id += 1
        self._seg_start = start_ms
        self._seg_frames = sum(1 for f in self._buffer if f.t_ms >= start_ms)
        self._next_hop = start_ms + self.cfg.hop_ms
        self.stats.segments_opened += 1

    def _slice(self, start_ms: int, end_ms: int) -> tuple[list[_Frame], int]:
        frames = [f for f in self._buffer if start_ms <= f.t_ms < end_ms]
        # The first frame's credit is time before the segment began.
        return frames, sum(f.signing_credit_ms for f in frames[1:])

    def _gate(self, duration_ms: int, n_frames: int, signing_ms: int) -> DropReason | None:
        """Why this run of frames must not reach the model, or None if it may."""
        if n_frames == 0:
            return "empty"
        if duration_ms < self.cfg.min_segment_ms:
            return "too_short"
        if signing_ms < self.cfg.min_signing_ms:
            return "no_signing"
        return None

    def _job(
        self,
        kind: Literal["partial", "final"],
        frames: list[_Frame],
        end_ms: int,
        signing_ms: int,
        closed_by: ClosedBy | None,
    ) -> DecodeJob:
        assert self._seg_id is not None
        return DecodeJob(
            kind=kind,
            segment_id=self._seg_id,
            start_ms=self._seg_start,
            end_ms=end_ms,
            timestamps_ms=np.array([f.t_ms for f in frames], dtype=np.int64),
            keypoints=np.stack([f.keypoints for f in frames]),
            scores=np.stack([f.scores for f in frames]),
            frame_size=self.frame_size,
            signing_ms=signing_ms,
            closed_by=closed_by,
        )

    def _close(self, end_ms: int, closed_by: ClosedBy, events: list[StreamEvent]) -> None:
        assert self._seg_id is not None
        stats = self.stats
        setattr(
            stats,
            f"segments_closed_by_{closed_by}",
            getattr(stats, f"segments_closed_by_{closed_by}") + 1,
        )
        frames, signing_ms = self._slice(self._seg_start, end_ms)
        reason = self._gate(end_ms - self._seg_start, len(frames), signing_ms)
        if reason is None:
            self._finals.append(self._job("final", frames, end_ms, signing_ms, closed_by))
            stats.finals_queued += 1
            if len(self._finals) > self.cfg.max_pending_finals:
                oldest = self._finals.popleft()
                stats.finals_dropped_backlog += 1
                events.append(self._dropped(oldest, "backlog"))
            stats.max_pending_finals = max(stats.max_pending_finals, len(self._finals))
        else:
            setattr(
                stats,
                f"segments_dropped_{reason}",
                getattr(stats, f"segments_dropped_{reason}") + 1,
            )
            events.append(
                StreamEvent(
                    type="dropped",
                    segment_id=self._seg_id,
                    text="",
                    start_ms=self._seg_start,
                    end_ms=end_ms,
                    n_frames=len(frames),
                    signing_ms=signing_ms,
                    closed_by=closed_by,
                    reason=reason,
                )
            )
        if self._partial is not None:  # a partial of a closed segment is never decoded
            self._partial = None
            stats.partials_replaced_before_decoding += 1
        self._seg_id = None
        self._seg_frames = 0
        self._last_end = end_ms

    @staticmethod
    def _dropped(job: DecodeJob, reason: DropReason) -> StreamEvent:
        return StreamEvent(
            type="dropped",
            segment_id=job.segment_id,
            text="",
            start_ms=job.start_ms,
            end_ms=job.end_ms,
            n_frames=job.n_frames,
            signing_ms=job.signing_ms,
            closed_by=job.closed_by,
            reason=reason,
        )

    def _hop(self, t: int) -> None:
        stats = self.stats
        stats.hops_due += 1
        hop = self.cfg.hop_ms
        self._next_hop += hop * ((t - self._next_hop) // hop + 1)  # no catching up in bursts
        if self.busy:
            stats.hops_skipped_busy += 1
            return
        frames, signing_ms = self._slice(self._seg_start, t + 1)
        if self._gate(t - self._seg_start, len(frames), signing_ms) is not None:
            stats.hops_skipped_no_signing += 1
            return
        if self._partial is not None:
            stats.partials_replaced_before_decoding += 1
        self._partial = self._job("partial", frames, t + 1, signing_ms, None)
        stats.partials_requested += 1

    def _trim(self, t: int) -> None:
        if self._seg_id is not None:
            keep_from = self._seg_start
        else:
            keep_from = t - self.detector.lookback_ms
            if self._last_end is not None:
                keep_from = max(keep_from, self._last_end)
        buffer = self._buffer
        while buffer and (buffer[0].t_ms < keep_from or len(buffer) > self.max_frames + 1):
            buffer.popleft()


# --- decoding with the real model ----------------------------------------------------------------


def offline_decoder(
    offline: OfflineTranslator, decode: DecodeConfig | None = None
) -> Callable[[DecodeJob], str]:
    """Translate a job through the offline path (`translate_arrays`): the same thinning
    rule, the same normalization over the segment, the same decoder."""

    def run(job: DecodeJob) -> str:
        return offline.translate_arrays(
            job.keypoints,
            job.scores,
            job.frame_size,
            clip_id=f"segment-{job.segment_id}-{job.kind}",
            keypoints_from="stream",
            decode=decode,
        ).text

    return run


# --- replay ------------------------------------------------------------------------------------


def keypoints_for_replay(
    video: str | Path, paths: PathsConfig, keypoint_cfg: KeypointConfig
) -> tuple[RawKeypoints, str]:
    """A video's raw keypoints through the keypoint cache, extracting them on a miss (the
    pose models load only then). Returns them with "cached" or "extracted"."""
    from signtoenglish.preprocess.extract import PoseCore

    entry, outcome = get_or_extract(
        Source(path=Path(video), expected_sha256=None, origin="replay"),
        KeypointCache(paths.cache_root, keypoint_cfg),
        lambda: PoseCore(keypoint_cfg, paths.assets_root),
    )
    if entry is None:
        raise StreamError(outcome.detail)
    return entry.raw, outcome.status


FrameTuple = tuple[int, npt.NDArray[np.float32], npt.NDArray[np.float32], bool]


def iter_raw(raw: RawKeypoints, loop_to_ms: int | None = None) -> Iterator[FrameTuple]:
    """The frames of cached keypoints, in order, with their own timestamps.

    With `loop_to_ms`, the recording is repeated until that much stream time has passed.
    Each repeat is shifted by the recording's length plus its own median frame interval, so
    timestamps keep increasing and no rate is assumed.
    """
    n = len(raw)
    if n == 0:
        return
    ts = raw.timestamps_ms
    first = int(ts[0])
    period = int(ts[-1]) - first + (int(np.median(np.diff(ts))) if n > 1 else 1)
    shift = 0
    while True:
        for i in range(n):
            t = int(ts[i]) + shift
            if loop_to_ms is not None and t - first >= loop_to_ms:
                return
            yield t, raw.keypoints[i], raw.scores[i], bool(raw.person_found[i])
        if loop_to_ms is None:
            return
        shift += period


@dataclass(slots=True)
class ReplayResult:
    events: list[StreamEvent]
    stats: StreamStats
    pace: Pace
    #: "real" when a translate function ran every job; "disabled" when there was none.
    decoding: Literal["real", "disabled"]
    first_ms: int | None = None
    last_ms: int | None = None
    partial_decode_ms: list[float] = field(default_factory=list)
    final_decode_ms: list[float] = field(default_factory=list)
    #: Stream time between consecutive text updates (partial or final) of one segment.
    update_intervals_ms: list[float] = field(default_factory=list)
    #: Stream time from each committed segment's end to its text (empty in "instant" pace).
    commit_delays_ms: list[float] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def finals(self) -> list[StreamEvent]:
        return [e for e in self.events if e.type == "final"]

    @property
    def stream_ms(self) -> int:
        if self.first_ms is None or self.last_ms is None:
            return 0
        return self.last_ms - self.first_ms

    def timing_summary(self) -> dict[str, Any]:
        """Translation-stage timing only: replay starts from keypoints, so frame decoding
        and pose estimation are not in it."""

        def spread(values: list[float]) -> dict[str, Any] | None:
            if not values:
                return None
            arr = np.asarray(values, dtype=np.float64)
            return {
                "n": len(values),
                "median": round(float(np.median(arr)), 1),
                "p90": round(float(np.percentile(arr, 90)), 1),
                "max": round(float(arr.max()), 1),
            }

        decode_total = sum(self.partial_decode_ms) + sum(self.final_decode_ms)
        return {
            "pace": self.pace,
            "decoding": self.decoding,
            "stream_s": round(self.stream_ms / 1000, 2),
            "wall_s": round(self.wall_s, 2),
            "decode_total_s": round(decode_total / 1000, 2),
            "translation_real_time_factor": round(decode_total / self.stream_ms, 3)
            if self.stream_ms
            else None,
            "partial_decode_ms": spread(self.partial_decode_ms),
            "final_decode_ms": spread(self.final_decode_ms),
            "update_interval_ms": spread(self.update_intervals_ms),
            # None in "instant" pace, where text is taken to arrive at once.
            "commit_delay_ms": spread(self.commit_delays_ms),
        }


def replay(
    engine: StreamingEngine,
    frames: Iterable[FrameTuple],
    translate: Callable[[DecodeJob], str] | None,
    *,
    pace: Pace = "simulated",
    assumed_decode_ms: float = 0.0,
    on_event: Callable[[StreamEvent], None] | None = None,
    on_frame: Callable[[int], None] | None = None,
    keep_events: bool = True,
) -> ReplayResult:
    """Drive an engine from recorded keypoints, with one decoder, in this thread.

    `pace`:
    - "instant": every job is translated before the next frame is pushed, as if decoding
      took no time. Every hop gets its partial. Deterministic; used to compare with offline.
    - "simulated": stream time is the clock. A job started at stream time `t` that took `d`
      of real time delivers its text at `t + d`; frames stamped before that are pushed
      first, so the engine sees a busy decoder and skips hops exactly as it would live.
      Nothing sleeps, so a replay takes about as long as its decoding.
    - "realtime": as "simulated", and frames are also held back to their own timestamps on
      the wall clock, so text appears at the speed of the recording.

    `translate=None` disables decoding: jobs come back with empty text after
    `assumed_decode_ms` of stream time. That exercises the buffer and the scheduling only;
    the result says `decoding="disabled"`, and it is not a test of the model.

    `keep_events=False` hands events to `on_event` only, for replays too long to keep.
    """
    result = ReplayResult(
        events=[],
        stats=engine.stats,
        pace=pace,
        decoding="real" if translate is not None else "disabled",
    )
    last_update: dict[int, float] = {}
    out: tuple[DecodeJob, str | None, str, float, float] | None = None
    wall_start = time.perf_counter()

    def emit(events: list[StreamEvent], at_ms: float) -> None:
        for event in events:
            if keep_events:
                result.events.append(event)
            if event.type in ("partial", "final"):
                previous = last_update.get(event.segment_id)
                if previous is not None and pace != "instant":
                    result.update_intervals_ms.append(at_ms - previous)
                last_update[event.segment_id] = at_ms
                if event.type == "final":
                    last_update.pop(event.segment_id, None)
                    if event.latency_ms is not None:
                        result.commit_delays_ms.append(event.latency_ms)
            if on_event is not None:
                on_event(event)

    def run(job: DecodeJob) -> tuple[str | None, str, float]:
        """(text, error detail, decode milliseconds)."""
        if translate is None:
            return "", "", float(assumed_decode_ms)
        started = time.perf_counter()
        try:
            text: str | None = translate(job)
            detail = ""
        except Exception as exc:  # one bad segment must not end the stream
            text, detail = None, f"{type(exc).__name__}: {exc}"
        elapsed = 1000 * (time.perf_counter() - started)
        (result.final_decode_ms if job.kind == "final" else result.partial_decode_ms).append(
            elapsed
        )
        return text, detail, elapsed

    def hand_back(
        job: DecodeJob, text: str | None, detail: str, ms: float, at: float | None
    ) -> None:
        when = None if at is None else round(at)
        if text is None:
            emit(engine.fail(job, detail), at or 0.0)
        else:
            emit(engine.complete(job, text, ms, now_ms=when), at or 0.0)

    def start(now: float) -> None:
        nonlocal out
        if out is None:
            job = engine.next_job()
            if job is not None:
                text, detail, ms = run(job)
                out = (job, text, detail, ms, now + ms)

    def finish_until(now: float | None) -> None:
        """Deliver every job whose text is ready by `now` (all of them if None)."""
        nonlocal out
        while out is not None and (now is None or out[4] <= now):
            job, text, detail, ms, done_at = out
            out = None
            hand_back(job, text, detail, ms, done_at)
            start(done_at)

    def drain_instant() -> None:
        while (job := engine.next_job()) is not None:
            text, detail, ms = run(job)
            hand_back(job, text, detail, ms, None)

    for t_ms, keypoints, scores, person_found in frames:
        if result.first_ms is None:
            result.first_ms = t_ms
        result.last_ms = t_ms if result.last_ms is None else max(result.last_ms, t_ms)
        if pace == "instant":
            emit(engine.push(t_ms, keypoints, scores, person_found), t_ms)
            drain_instant()
        else:
            finish_until(t_ms)
            if pace == "realtime":
                wait = (t_ms - result.first_ms) / 1000 - (time.perf_counter() - wall_start)
                if wait > 0:
                    time.sleep(wait)
            emit(engine.push(t_ms, keypoints, scores, person_found), t_ms)
            start(t_ms)
        if on_frame is not None:
            on_frame(t_ms)

    end = float(result.last_ms) if result.last_ms is not None else 0.0
    emit(engine.flush(), end)
    if pace == "instant":
        drain_instant()
    else:
        start(end)
        finish_until(None)
    result.wall_s = time.perf_counter() - wall_start
    return result


# --- memory, for the bounded-memory check -----------------------------------------------------


def memory_snapshot() -> dict[str, float]:
    """Megabytes in use by this process right now: resident memory as the system reports
    it, and what PyTorch holds on the MPS device (not all of which is resident memory)."""
    import os

    out: dict[str, float] = {}
    try:
        done = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        out["rss_mb"] = round(int(done.stdout.strip()) / 1024, 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        import torch

        if torch.backends.mps.is_available():
            out["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 2**20, 1)
            out["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2**20, 1)
    except ImportError:
        pass
    return out
