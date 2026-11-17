"""The streaming engine's logic on synthetic keypoint frames: buffer, timestamps, commit
rules, the no-signing guard, forced cuts, hop scheduling, and bounded memory.

No model here. Where a test needs text it uses `label`, which names the job it was given;
that is enough to check WHICH frames would be translated and WHEN. Whether the text is
right is tested with the real model in tests/infer/test_streaming_model.py.
"""

from __future__ import annotations

import itertools
import threading
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signtoenglish.common.config import ConfigError
from signtoenglish.infer.streaming import (
    COMMITTED_HISTORY,
    DecodeJob,
    StreamConfig,
    StreamError,
    StreamEvent,
    StreamingEngine,
    iter_raw,
    load_stream_config,
    replay,
)
from signtoenglish.models.boundary import (
    BoundaryDetector,
    BoundaryEvent,
    BoundaryUpdate,
    Observation,
)
from signtoenglish.preprocess.schema import RawKeypoints

Frames = Callable[..., list[tuple[int, np.ndarray, np.ndarray, bool]]]
Engine = Callable[..., StreamingEngine]
FIXED_2S = {"kind": "fixed_length", "length_ms": 2000}


def _push_all(engine: StreamingEngine, frames: list[Any]) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for t, keypoints, scores, found in frames:
        events += engine.push(t, keypoints, scores, found)
    return events


def _irregular_times(seconds: float, seed: int = 0) -> list[int]:
    """Frame times with intervals anywhere from 20 to 90 ms: no rate to assume."""
    rng = np.random.default_rng(seed)
    out, t = [], 17
    while t < seconds * 1000:
        out.append(t)
        t += int(rng.integers(20, 91))
    return out


# --- config --------------------------------------------------------------------------------


def test_default_config_file_matches_the_code_defaults(repo_root: Path, tmp_path: Path) -> None:
    assert load_stream_config(repo_root / "configs" / "stream" / "default.yaml") == StreamConfig()
    bad = tmp_path / "bad.yaml"
    bad.write_text("hop_ms: 1000\nhop_seconds: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_stream_config(bad)
    bad.write_text("boundary:\n  kind: pause_splitter\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_stream_config(bad)


# --- segments, partials, finals --------------------------------------------------------------


def test_fixed_length_gives_partials_every_hop_and_one_final_per_segment(
    engine: Engine, frames: Frames, times: Callable[..., list[int]], label: Any
) -> None:
    eng = engine(boundary=FIXED_2S, hop_ms=500, min_segment_ms=500, min_signing_ms=300)
    result = replay(eng, frames(times(5.0)), label, pace="instant")
    assert [(e.type, e.text) for e in result.events] == [
        ("partial", "partial 0-501 n=16"),
        ("partial", "partial 0-1001 n=31"),
        ("partial", "partial 0-1501 n=46"),
        ("final", "final 0-2000 n=60"),
        ("partial", "partial 2000-2501 n=16"),
        ("partial", "partial 2000-3001 n=31"),
        ("partial", "partial 2000-3501 n=46"),
        ("final", "final 2000-4000 n=60"),
        ("partial", "partial 4000-4501 n=16"),
        ("final", "final 4000-4968 n=30"),
    ]
    finals = result.finals
    assert [e.segment_id for e in finals] == [0, 1, 2]
    assert [e.closed_by for e in finals] == ["boundary", "boundary", "flush"]
    assert eng.committed == tuple(finals)
    s = eng.stats
    assert (s.segments_opened, s.finals_queued, s.finals_emitted) == (3, 3, 3)
    assert s.hops_due == s.partials_requested == s.partials_emitted == 7
    assert s.hops_skipped_busy == 0


def test_boundaries_follow_timestamps_not_frame_counts(
    engine: Engine, frames: Frames, label: Any
) -> None:
    ts = _irregular_times(12.0)
    seen: list[DecodeJob] = []

    def translate(job: DecodeJob) -> str:
        seen.append(job)
        return label(job)

    eng = engine(boundary={"kind": "fixed_length", "length_ms": 3000})
    result = replay(eng, frames(ts), translate, pace="instant")
    finals = [j for j in seen if j.kind == "final"]
    assert len(finals) >= 3
    arr = np.array(ts)
    previous_end = ts[0]
    for job in finals:
        assert job.start_ms == previous_end  # segments tile the stream
        if job.closed_by == "boundary":
            assert job.end_ms - job.start_ms >= 3000
            # ... and the cut came at the first frame at or past the length
            assert arr[arr < job.end_ms].max() - job.start_ms < 3000
        expected = arr[(arr >= job.start_ms) & (arr < job.end_ms)]
        assert job.timestamps_ms.tolist() == expected.tolist()
        previous_end = job.end_ms
    for job in seen:
        if job.kind == "partial":
            expected = arr[(arr >= job.start_ms) & (arr < job.end_ms)]
            assert job.timestamps_ms.tolist() == expected.tolist()
            assert job.end_ms == job.timestamps_ms[-1] + 1
    assert len(result.finals) == len(finals)


def test_oracle_segments_are_the_offline_slice(engine: Engine, frames: Frames) -> None:
    """A streamed segment holds exactly the frames with start <= t < end, which is the
    rule `OfflineTranslator.translate_raw` cuts by."""
    ts = _irregular_times(12.0, seed=3)
    windows = [(1000, 3500), (3500, 6000), (7000, 9000), (9500, 20000)]
    seen: list[DecodeJob] = []

    def translate(job: DecodeJob) -> str:
        if job.kind == "final":
            seen.append(job)
        return ""

    eng = engine(boundary={"kind": "oracle", "segments": windows}, min_segment_ms=0)
    replay(eng, frames(ts), translate, pace="instant")
    arr = np.array(ts)
    assert [(j.start_ms, j.end_ms) for j in seen[:3]] == windows[:3]
    for job, (a, b) in zip(seen, windows, strict=True):
        assert job.timestamps_ms.tolist() == arr[(arr >= a) & (arr < b)].tolist()
    # The last window runs past the end of the stream: closed by the flush, same frames.
    assert seen[3].closed_by == "flush" and seen[3].end_ms == ts[-1] + 1
    assert eng.stats.boundary_events_ignored == 0


def test_jobs_hold_copies_of_the_frames(engine: Engine, make_frame: Any) -> None:
    eng = engine(boundary=FIXED_2S)
    keypoints, scores = make_frame()
    for t in range(0, 1100, 50):
        keypoints[:] = t / 10  # the caller reuses its arrays
        eng.push(t, keypoints, scores)
    job = eng.next_job()
    assert job is not None and job.kind == "partial"
    assert job.keypoints[:, 0, 0].tolist() == [t / 10 for t in range(0, 1050, 50)]
    assert job.keypoints.dtype == np.float32 and job.scores.shape == (21, 133)
    assert job.frame_size == (640, 480)


def test_bad_input_is_refused(engine: Engine, make_frame: Any) -> None:
    eng = engine()
    keypoints, scores = make_frame()
    with pytest.raises(StreamError, match="expected keypoints"):
        eng.push(0, keypoints[:100], scores)
    with pytest.raises(StreamError, match="max_frames"):
        StreamingEngine(StreamConfig(), (640, 480), 0)
    with pytest.raises(StreamError, match="frame size"):
        StreamingEngine(StreamConfig(), (0, 480), 256)


# --- committed text never changes ------------------------------------------------------------


def test_a_partial_that_returns_after_its_segment_closed_is_discarded(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    eng = engine(boundary=FIXED_2S)
    all_frames = frames(times(2.5))
    _push_all(eng, [f for f in all_frames if f[0] <= 1000])
    partial = eng.next_job()
    assert partial is not None and partial.kind == "partial" and partial.segment_id == 0
    _push_all(eng, [f for f in all_frames if f[0] > 1000])  # segment 0 closes at 2000
    assert eng.next_job() is None  # one job out at a time
    assert eng.complete(partial, "late partial", 5.0) == []
    assert eng.stats.partials_discarded_stale == 1
    final = eng.next_job()
    assert final is not None and final.kind == "final" and final.segment_id == 0
    (event,) = eng.complete(final, "the sentence", 5.0, now_ms=2400)
    assert (event.type, event.text, event.latency_ms) == ("final", "the sentence", 400.0)
    with pytest.raises(StreamError, match="not the one that is out"):
        eng.complete(final, "again", 5.0)
    assert [e.text for e in eng.committed] == ["the sentence"]


def test_no_partial_follows_a_final_and_no_segment_commits_twice(
    engine: Engine, frames: Frames, times: Callable[..., list[int]], label: Any
) -> None:
    for pace, assumed in (("instant", 0.0), ("simulated", 300.0), ("simulated", 1700.0)):
        eng = engine(boundary=FIXED_2S, hop_ms=400, min_segment_ms=400, min_signing_ms=300)
        result = replay(
            eng,
            frames(times(21.0)),
            None if assumed else label,
            pace=pace,  # type: ignore[arg-type]
            assumed_decode_ms=assumed,
        )
        committed: set[int] = set()
        for event in result.events:
            assert event.segment_id not in committed, (pace, assumed, event)
            if event.type == "final":
                committed.add(event.segment_id)
        assert len(committed) == len(result.finals) == 11
        assert [e.segment_id for e in result.finals] == sorted(committed)


def test_the_engine_keeps_a_bounded_history_of_committed_text(
    engine: Engine, frames: Frames, times: Callable[..., list[int]], label: Any
) -> None:
    eng = engine(boundary={"kind": "fixed_length", "length_ms": 1000})
    n = COMMITTED_HISTORY + 20
    result = replay(eng, frames(times(n + 0.5)), label, pace="instant")
    assert len(result.finals) == n
    assert eng.committed == tuple(result.finals[-COMMITTED_HISTORY:])


# --- nothing without signing reaches the model ----------------------------------------------------


def _replay_counting(eng: StreamingEngine, frames: list[Any]) -> tuple[list[DecodeJob], Any]:
    jobs: list[DecodeJob] = []

    def translate(job: DecodeJob) -> str:
        jobs.append(job)
        return "text"

    return jobs, replay(eng, frames, translate, pace="instant")


@pytest.mark.parametrize("case", ["low_score", "just_above_the_model_mask", "outside_the_frame"])
def test_a_stream_without_hands_never_produces_a_job(
    engine: Engine, times: Callable[..., list[int]], make_frame: Any, case: str
) -> None:
    keypoints, scores = make_frame(score={"low_score": 0.1}.get(case, 0.35))
    if case == "outside_the_frame":
        keypoints, scores = make_frame(score=0.9)
        keypoints[91:133] = (-40.0, 500.0)
    eng = engine(boundary=FIXED_2S)
    jobs, result = _replay_counting(eng, [(t, keypoints, scores, True) for t in times(9.0)])
    assert jobs == []
    assert [e.type for e in result.events] == ["dropped"] * 5
    assert [e.reason for e in result.events] == ["no_signing"] * 4 + ["too_short"]
    s = eng.stats
    assert (s.segments_dropped_no_signing, s.segments_dropped_too_short) == (4, 1)
    assert s.partials_requested == 0 and s.hops_skipped_no_signing == s.hops_due > 0
    assert not eng.signing


def test_the_guard_counts_signing_time_inside_each_segment(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    # Hands up for 0.3 s of the first window, 0.7 s of the second, none of the third.
    def hands(t: int) -> bool:
        return 500 <= t < 800 or 2500 <= t < 3200

    eng = engine(boundary=FIXED_2S)
    jobs, result = _replay_counting(eng, frames(times(6.1), hands))
    finals = [j for j in jobs if j.kind == "final"]
    assert [(j.start_ms, j.end_ms) for j in finals] == [(2000, 4000)]
    assert 650 <= finals[0].signing_ms <= 750
    # Partials of the second window start only once it holds enough signing.
    partials = [j for j in jobs if j.kind == "partial"]
    assert partials and all(j.start_ms == 2000 and j.signing_ms >= 500 for j in partials)
    assert partials[0].end_ms > 3000
    dropped = [(e.start_ms, e.reason) for e in result.events if e.type == "dropped"]
    assert dropped == [(0, "no_signing"), (4000, "no_signing"), (6000, "too_short")]


class _ClaimsSegmentsButNoSigning(BoundaryDetector):
    """A detector that opens and closes segments while saying no frame shows signing."""

    def update(self, obs: Observation) -> BoundaryUpdate:
        if obs.t_ms % 3000 < 20:
            return BoundaryUpdate(
                False, (BoundaryEvent("end", obs.t_ms), BoundaryEvent("start", obs.t_ms))
            )
        return BoundaryUpdate(False)


def test_a_detector_cannot_route_around_the_guard(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    eng = engine(detector=_ClaimsSegmentsButNoSigning())
    jobs, result = _replay_counting(eng, frames(times(9.0), hands=True))
    assert jobs == [] and eng.stats.segments_opened == 3
    assert {e.reason for e in result.events} == {"no_signing"}
    with pytest.raises(Exception, match="greater than 0"):
        StreamConfig(min_signing_ms=0)  # the guard cannot be configured away either


def test_short_segments_are_dropped_without_decoding(engine: Engine, frames: Frames) -> None:
    eng = engine(boundary={"kind": "oracle", "segments": [(0, 900), (2000, 3000)]})
    jobs, result = _replay_counting(eng, frames(range(0, 4000, 40)))
    assert [(j.kind, j.start_ms, j.end_ms) for j in jobs] == [("final", 2000, 3000)]
    assert next((e.type, e.reason) for e in result.events) == ("dropped", "too_short")


# --- the base model's maximum ----------------------------------------------------------------


def test_a_segment_is_cut_at_the_models_maximum_frame_count(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    ts = times(10.0)
    eng = engine(max_frames=100, boundary={"kind": "oracle", "segments": [(0, 60000)]})
    jobs, _ = _replay_counting(eng, frames(ts))
    finals = [j for j in jobs if j.kind == "final"]
    assert [j.n_frames for j in finals] == [100, 100, 100]
    assert [j.closed_by for j in finals] == ["max_length", "max_length", "flush"]
    assert [j.start_ms for j in finals] == [ts[0], ts[100], ts[200]]
    assert [j.end_ms for j in finals] == [ts[100], ts[200], ts[299] + 1]
    assert all(j.n_frames <= 100 for j in jobs)
    assert eng.stats.segments_closed_by_max_length == 2
    assert eng.stats.max_buffered_frames <= 101


def test_a_segment_is_cut_at_the_maximum_duration(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    eng = engine(boundary={"kind": "fixed_length", "length_ms": 8000}, max_segment_ms=3000)
    jobs, _ = _replay_counting(eng, frames(times(9.5)))
    finals = [j for j in jobs if j.kind == "final"]
    assert [(j.start_ms, j.end_ms, j.closed_by) for j in finals] == [
        (0, 3000, "max_length"),
        (3000, 6000, "max_length"),
        (6000, 9000, "max_length"),
    ]


# --- timestamps: duplicates, out of order, gaps ----------------------------------------------


def test_duplicate_and_out_of_order_timestamps_are_dropped_and_counted(
    engine: Engine, make_frame: Any
) -> None:
    eng = engine(boundary=FIXED_2S)
    keypoints, scores = make_frame()
    clean = list(range(0, 3000, 40))
    noisy: list[int] = []
    for t in clean:
        noisy.append(t)
        if t % 400 == 0:
            noisy.append(t)  # the same frame again
        if t % 1000 == 0 and t > 0:
            noisy += [t - 120, t - 40]  # two frames from the past
    for t in noisy:
        eng.push(t, keypoints, scores)
    s = eng.stats
    assert s.frames_pushed == len(noisy)
    assert s.duplicate_timestamps == 8 and s.out_of_order_timestamps == 4
    assert s.frames_accepted == len(clean) and s.gaps == 0
    job = eng.next_job()
    assert job is not None and job.kind == "final"
    assert job.timestamps_ms.tolist() == [t for t in clean if t < 2000]
    assert job.signing_ms == 1960  # the dropped frames added no signing time


def test_a_gap_closes_the_open_segment_at_the_last_frame_before_it(
    engine: Engine, frames: Frames
) -> None:
    before = list(range(0, 1600, 40))  # last frame at 1560
    after = list(range(5000, 7400, 40))
    eng = engine(boundary=FIXED_2S)
    jobs, _ = _replay_counting(eng, frames(before + after))
    finals = [j for j in jobs if j.kind == "final"]
    assert [(j.start_ms, j.end_ms, j.closed_by) for j in finals] == [
        (0, 1561, "gap"),
        (5000, 7000, "boundary"),
    ]
    # No signing time is credited across the gap, and no segment spans it.
    assert finals[1].signing_ms == 1960 and finals[1].timestamps_ms[0] == 5000
    s = eng.stats
    assert (s.gaps, s.segments_closed_by_gap) == (1, 1)
    # Partials after the gap count from the new segment's start.
    late = [j for j in jobs if j.kind == "partial" and j.start_ms == 5000]
    assert [j.end_ms for j in late] == [6001]


def test_a_gap_inside_an_oracle_window_gives_two_segments(engine: Engine, frames: Frames) -> None:
    ts = list(range(0, 3000, 40)) + list(range(4200, 8000, 40))
    eng = engine(boundary={"kind": "oracle", "segments": [(1000, 7000)]})
    jobs, _ = _replay_counting(eng, frames(ts))
    finals = [j for j in jobs if j.kind == "final"]
    assert [(j.start_ms, j.end_ms, j.closed_by) for j in finals] == [
        (1000, 2961, "gap"),
        (4200, 7000, "boundary"),
    ]


def test_a_boundary_event_from_the_future_is_an_error(engine: Engine, make_frame: Any) -> None:
    class Ahead(BoundaryDetector):
        def update(self, obs: Observation) -> BoundaryUpdate:
            return BoundaryUpdate(True, (BoundaryEvent("start", obs.t_ms + 1),))

    eng = engine(detector=Ahead())
    with pytest.raises(StreamError, match="after the newest frame"):
        eng.push(0, *make_frame())


class _LooksBack(BoundaryDetector):
    """Reports a start 300 ms late and an end 300 ms late, as a detector with look-ahead does."""

    lookback_ms = 300

    def update(self, obs: Observation) -> BoundaryUpdate:
        events = []
        if obs.t_ms == 1300:
            events.append(BoundaryEvent("start", 1000))
        if obs.t_ms == 4300:
            events.append(BoundaryEvent("end", 4000))
        return BoundaryUpdate(True, tuple(events))


def test_events_may_lie_in_the_past_by_the_detectors_lookback(
    engine: Engine, frames: Frames
) -> None:
    eng = engine(detector=_LooksBack())
    jobs, _ = _replay_counting(eng, frames(range(0, 6000, 50)))
    (final,) = [j for j in jobs if j.kind == "final"]
    assert (final.start_ms, final.end_ms) == (1000, 4000)
    assert final.timestamps_ms.tolist() == list(range(1000, 4000, 50))
    assert eng.stats.max_buffered_frames <= 4300 // 50  # idle history is the look-back only


# --- hop scheduling with a slow decoder ----------------------------------------------------------


def test_a_hop_that_comes_due_while_a_job_is_out_is_skipped(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    eng = engine(boundary={"kind": "fixed_length", "length_ms": 8000})
    by_second: dict[int, list[Any]] = {}
    for f in frames(times(9.0)):
        by_second.setdefault(f[0] // 1000, []).append(f)

    _push_all(eng, by_second[0] + by_second[1][:1])  # up to t = 1000: first hop
    job = eng.next_job()
    assert job is not None and (job.kind, job.end_ms) == ("partial", 1001)
    _push_all(eng, by_second[1][1:] + by_second[2][:1])  # hop at 2000 while the job is out
    assert eng.stats.hops_skipped_busy == 1 and eng.next_job() is None
    (event,) = eng.complete(job, "one", 12.0, now_ms=2100)
    assert (event.type, event.end_ms, event.latency_ms) == ("partial", 1001, 1099.0)

    _push_all(eng, by_second[2][1:] + by_second[3][:1])  # hop at 3000: requested, not taken
    _push_all(eng, by_second[3][1:] + by_second[4][:1])  # hop at 4000 replaces it
    assert eng.stats.partials_requested == 3
    assert eng.stats.partials_replaced_before_decoding == 1
    job = eng.next_job()
    assert job is not None and job.end_ms == 4001  # the newest request, not the stale one
    eng.complete(job, "two", 12.0)

    for second in (4, 5, 6, 7):
        _push_all(eng, by_second[second][1:] + by_second[second + 1][:1])
    # The segment closed at 8000 with a partial request pending: the final goes first and
    # the pending partial is never decoded.
    job = eng.next_job()
    assert job is not None and (job.kind, job.start_ms, job.end_ms) == ("final", 0, 8000)
    assert eng.busy
    eng.complete(job, "sentence", 12.0)
    assert eng.next_job() is None and not eng.busy


def test_simulated_pace_skips_hops_and_measures_delays_on_the_stream_clock(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    def run() -> Any:
        eng = engine(boundary={"kind": "fixed_length", "length_ms": 8000})
        return replay(eng, frames(times(24.5)), None, pace="simulated", assumed_decode_ms=1500.0)

    result = run()
    s = result.stats
    assert result.decoding == "disabled"
    assert s.hops_skipped_busy > 0
    assert s.hops_due == s.hops_skipped_busy + s.partials_requested
    assert len(result.finals) == 3
    # A decode lasts 1.5 s of stream time, so text cannot update faster than that ...
    assert min(result.update_intervals_ms) >= 1500
    # ... and a final waits for the partial that is out, then for its own decode.
    assert result.commit_delays_ms == [e.latency_ms for e in result.finals]
    assert all(1500 <= d <= 3000 for d in result.commit_delays_ms)
    summary = result.timing_summary()
    assert summary["commit_delay_ms"]["n"] == 3 and summary["decoding"] == "disabled"
    again = run()
    assert [e.to_dict() for e in again.events] == [e.to_dict() for e in result.events]


def test_instant_pace_reports_no_latency(
    engine: Engine, frames: Frames, times: Callable[..., list[int]], label: Any
) -> None:
    result = replay(engine(boundary=FIXED_2S), frames(times(4.5)), label, pace="instant")
    assert all(e.latency_ms is None for e in result.events)
    assert result.timing_summary()["commit_delay_ms"] is None
    assert result.decoding == "real" and len(result.final_decode_ms) == 2


def test_realtime_pace_holds_frames_to_their_timestamps(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    started = time.perf_counter()
    replay(engine(), frames(times(0.4)), None, pace="realtime")
    assert time.perf_counter() - started >= 0.35  # a lower bound only: sleeping, not speed


def test_finals_wait_in_order_and_the_backlog_is_capped(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    eng = engine(boundary=FIXED_2S, max_pending_finals=3)
    events = _push_all(eng, frames(times(20.5)))  # ten segments, nobody decoding
    s = eng.stats
    assert s.finals_queued == 10 and s.finals_dropped_backlog == 7
    assert eng.pending_finals == 3 == s.max_pending_finals
    dropped = [e for e in events if e.type == "dropped"]
    assert [(e.segment_id, e.reason) for e in dropped] == [(i, "backlog") for i in range(7)]
    order = []
    while (job := eng.next_job()) is not None:
        order.append((job.kind, job.segment_id))
        eng.complete(job, "", 1.0)
    assert order == [("final", 7), ("final", 8), ("final", 9)]


def test_a_failed_decode_is_reported_and_the_stream_goes_on(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    def translate(job: DecodeJob) -> str:
        if job.kind == "final" and job.segment_id == 1:
            raise RuntimeError("decoder fell over")
        return f"ok {job.segment_id}"

    eng = engine(boundary=FIXED_2S)
    result = replay(eng, frames(times(6.5)), translate, pace="simulated")
    errors = [e for e in result.events if e.type == "error"]
    assert len(errors) == 1 and errors[0].segment_id == 1
    assert "decoder fell over" in (errors[0].reason or "")
    assert [e.text for e in result.finals] == ["ok 0", "ok 2"]
    assert eng.stats.decode_failures == 1 and not eng.busy


def test_one_receiver_and_one_decode_worker_can_share_an_engine(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    eng = engine(boundary={"kind": "fixed_length", "length_ms": 1000}, hop_ms=200, min_segment_ms=0)
    done = threading.Event()
    got: list[StreamEvent] = []

    def worker() -> None:
        while True:
            job = eng.next_job()
            if job is None:
                if done.is_set():
                    return
                time.sleep(0.0005)
                continue
            got.extend(eng.complete(job, f"{job.kind} {job.segment_id}", 0.1))

    thread = threading.Thread(target=worker)
    thread.start()
    dropped = _push_all(eng, frames(times(60.0)))
    dropped += eng.flush()
    done.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    finals = [e.segment_id for e in got if e.type == "final"]
    s = eng.stats
    assert finals == sorted(set(finals))
    assert len(finals) == s.finals_emitted == s.finals_queued - s.finals_dropped_backlog
    assert s.finals_queued + s.segments_dropped_too_short + s.segments_dropped_empty == 60
    committed: set[int] = set()
    for event in got:
        assert event.segment_id not in committed
        if event.type == "final":
            committed.add(event.segment_id)


# --- bounded memory --------------------------------------------------------------------------


def test_buffer_and_queues_stay_bounded_over_thirty_minutes_of_frames(
    engine: Engine, make_frame: Any
) -> None:
    """Thirty minutes of stream time through the buffer and the scheduling, with decoding
    DISABLED (no model). This bounds what the engine itself holds; the model's memory over
    a long replay is measured separately with the real model."""
    up, down = make_frame(True), make_frame(False)

    def stream() -> Any:
        t = 0
        rng = np.random.default_rng(1)
        while t < 30 * 60 * 1000:
            keypoints, scores = up if (t // 20000) % 3 else down  # 40 s of signing, 20 s of rest
            yield t, keypoints, scores, True
            t += int(rng.integers(25, 45))

    eng = engine(boundary={"kind": "fixed_length", "length_ms": 8000})
    marks: dict[int, int] = {}
    tracemalloc.start()
    try:

        def on_frame(t_ms: int) -> None:
            minute = t_ms // 60000
            if minute in (5, 29) and minute not in marks:
                marks[minute] = tracemalloc.get_traced_memory()[0]

        result = replay(
            eng,
            stream(),
            None,
            pace="simulated",
            assumed_decode_ms=900.0,
            on_frame=on_frame,
            keep_events=False,
        )
    finally:
        tracemalloc.stop()
    s = eng.stats
    assert result.stream_ms > 29.9 * 60 * 1000 and s.frames_accepted > 40000
    assert s.max_buffered_frames <= 257
    assert s.max_pending_finals <= eng.cfg.max_pending_finals
    assert s.finals_emitted > 100 and s.segments_dropped_no_signing > 10
    assert len(eng.committed) <= COMMITTED_HISTORY and result.events == []
    growth = marks[29] - marks[5]
    assert growth < 1_000_000, f"Python heap grew by {growth} bytes from minute 5 to minute 29"


def test_buffer_stays_bounded_when_nothing_is_ever_decoded(
    engine: Engine, frames: Frames, times: Callable[..., list[int]]
) -> None:
    eng = engine()
    _push_all(eng, frames(times(300.0, fps=60.0)))  # 480 frames per 8 s window: forced cuts
    s = eng.stats
    assert s.max_buffered_frames <= 257 and eng.pending_finals == eng.cfg.max_pending_finals
    assert s.segments_closed_by_max_length > 0 and s.finals_dropped_backlog > 0


# --- replaying cached keypoints ------------------------------------------------------------------


def test_iter_raw_keeps_real_timestamps_and_loops_without_assuming_a_rate() -> None:
    ts = np.array([40, 75, 110, 150, 181], dtype=np.int64)
    raw = RawKeypoints(
        timestamps_ms=ts,
        keypoints=np.zeros((5, 133, 2), np.float32),
        scores=np.zeros((5, 133), np.float32),
        person_found=np.ones(5, np.bool_),
        width=640,
        height=480,
    )
    assert [f[0] for f in iter_raw(raw)] == ts.tolist()
    looped = [f[0] for f in iter_raw(raw, loop_to_ms=400)]
    # Each repeat is shifted by the recording's span (141) plus its median interval (35).
    assert looped == [40, 75, 110, 150, 181, 216, 251, 286, 326, 357, 392, 427]
    assert all(b > a for a, b in itertools.pairwise(looped))
    empty = RawKeypoints(
        timestamps_ms=ts[:0],
        keypoints=np.zeros((0, 133, 2), np.float32),
        scores=np.zeros((0, 133), np.float32),
        person_found=np.ones(0, np.bool_),
        width=640,
        height=480,
    )
    assert list(iter_raw(empty, loop_to_ms=1000)) == []
