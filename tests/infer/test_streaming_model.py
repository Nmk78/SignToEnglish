"""The streaming engine with the real model and real keypoints (OpenASL checkpoint, CPU,
the instance shared with the other tests in this folder).

Replay material, all local and git-ignored: the two fixture clips, and a public-domain
102-second video of continuous ASL (`docs/DATA_LICENSES.md`). Its keypoints come from the
real extractor through the project's keypoint cache, so the first run extracts them (a few
minutes) and later runs read them.

There are no recorded continuous takes with marked sentence ends yet. The "oracle
boundaries" below are therefore ARBITRARY WINDOWS chosen for the test, not sentence
boundaries, and nothing here says anything about translation quality.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signtoenglish.cli import main
from signtoenglish.eval.stream_eval import oracle_check
from signtoenglish.infer import loader
from signtoenglish.infer.offline import OfflineTranslator
from signtoenglish.infer.streaming import (
    StreamConfig,
    StreamEvent,
    StreamingEngine,
    iter_raw,
    memory_snapshot,
    offline_decoder,
    replay,
)
from signtoenglish.models.boundary import FixedLengthConfig, OracleConfig
from signtoenglish.preprocess.cache import KeypointCache, Source, get_or_extract
from signtoenglish.preprocess.schema import RawKeypoints

CONTINUOUS = Path("commons_source") / "COVID-19_Prevention_Tips_in_ASL.webm"
#: Arbitrary windows (ms) over the continuous video: a 7 s one, one of 255 frames (just
#: under the model's maximum), and two back to back. Not sentence boundaries.
WINDOWS = ((2000, 9000), (12000, 20500), (30000, 33000), (33000, 38500))


@pytest.fixture(scope="module")
def offline(model_for: Callable[..., Any], paths: Any) -> OfflineTranslator:
    return OfflineTranslator(model_for("openasl", "cpu"), paths)


@pytest.fixture(scope="module")
def cached_raw(
    paths: Any, keypoint_cfg: Any, core: Any, clip_path: Callable[[str], Path], need_file: Any
) -> Callable[[str], RawKeypoints]:
    """Keypoints of a local clip through the project's keypoint cache."""
    cache = KeypointCache(paths.cache_root, keypoint_cfg)
    held: dict[str, RawKeypoints] = {}

    def get(name: str) -> RawKeypoints:
        if name not in held:
            if name == "continuous":
                folder = clip_path("hands_visible").parent
                path = need_file(folder / CONTINUOUS, "public-domain continuous ASL video")
            else:
                path = clip_path(name)
            entry, outcome = get_or_extract(Source(path, None, "test"), cache, lambda: core)
            assert entry is not None, outcome.detail
            held[name] = entry.raw
        return held[name]

    return get


def _check_commit_rules(events: list[StreamEvent]) -> None:
    """One final per segment, and nothing for a segment after its final."""
    committed: set[int] = set()
    for event in events:
        assert event.segment_id not in committed, event
        if event.type == "final":
            committed.add(event.segment_id)


# --- acceptance: with oracle boundaries, streaming equals offline, sentence for sentence -----


def test_oracle_boundaries_give_the_offline_text_window_for_window(
    offline: OfflineTranslator, cached_raw: Callable[[str], RawKeypoints]
) -> None:
    raw = cached_raw("continuous")
    assert len(raw) > 3000 and raw.timestamps_ms[-1] > 100_000
    check = oracle_check(offline, raw, WINDOWS, StreamConfig(hop_ms=2000))
    print("\noracle check (arbitrary windows, OpenASL checkpoint, CPU):")
    for w in check.windows:
        print(f"  {w.start_ms}-{w.end_ms} ms, {w.n_frames} frames, equal={w.equal}: "
              f"offline={w.offline_text!r} stream={w.stream_text!r} {w.stream_note}")  # fmt: skip
    assert check.all_equal, check.to_dict()
    assert [w.n_frames for w in check.windows][1] == 255  # close to the maximum, not thinned
    assert all(w.offline_text.strip() for w in check.windows)
    result = check.replay
    assert [(e.start_ms, e.end_ms) for e in result.finals] == list(WINDOWS)
    assert [e.closed_by for e in result.finals] == ["boundary"] * 4
    _check_commit_rules(result.events)
    s = result.stats
    assert s.finals_emitted == 4 and s.segments_closed_by_max_length == 0
    assert s.partials_emitted >= 8 and s.hops_skipped_busy == 0  # instant pace: every hop
    # Partial text is real model output on a growing slice of the same segment.
    partials = [e for e in result.events if e.type == "partial"]
    assert all(e.text.strip() and e.decode_ms and e.decode_ms > 0 for e in partials)
    first = [e for e in partials if e.segment_id == 0]
    assert [e.end_ms - e.start_ms for e in first] == sorted(e.end_ms - e.start_ms for e in first)


@pytest.mark.parametrize("stem", ["hands_visible", "hands_leave"])
def test_a_whole_clip_streamed_as_one_segment_equals_offline_translation_of_the_file(
    offline: OfflineTranslator,
    cached_raw: Callable[[str], RawKeypoints],
    clip_path: Callable[[str], Path],
    stem: str,
) -> None:
    raw = cached_raw(stem)
    expected = offline.translate_file(clip_path(stem))
    cfg = StreamConfig(boundary=OracleConfig(segments=((0, 60_000),)))
    engine = StreamingEngine(cfg, raw.frame_size, offline.loaded.translator.cfg.max_frames)
    result = replay(engine, iter_raw(raw), offline_decoder(offline), pace="instant")
    (final,) = result.finals
    assert final.text == expected.text and final.n_frames == expected.n_frames == len(raw)
    assert final.closed_by == "flush"


# --- nothing without signing reaches the model ---------------------------------------------------


def _spy(offline: OfflineTranslator, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the frames of every call that reaches the real translator."""
    calls: list[int] = []
    real = offline.loaded.translator.translate_with_ids

    def counted(clip: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(len(next(iter(clip.values()))))
        return real(clip, *args, **kwargs)

    monkeypatch.setattr(offline.loaded.translator, "translate_with_ids", counted)
    return calls


def test_the_part_of_a_clip_where_the_hands_have_left_never_reaches_the_model(
    offline: OfflineTranslator,
    cached_raw: Callable[[str], RawKeypoints],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`hands_leave`: hands in view for about 3 s, then out of the frame. The base model's
    own mask still passes one lowered hand in many of those frames; the engine's guard does
    not, and the model is never called for that stretch."""
    raw = cached_raw("hands_leave")
    calls = _spy(offline, monkeypatch)
    cfg = StreamConfig(boundary=OracleConfig(segments=((0, 3000), (3200, 4960))))
    engine = StreamingEngine(cfg, raw.frame_size, 256)
    result = replay(engine, iter_raw(raw), offline_decoder(offline), pace="instant")
    assert [(e.type, e.start_ms) for e in result.events if e.type != "partial"] == [
        ("final", 0),
        ("dropped", 3200),
    ]
    dropped = result.events[-1]
    assert dropped.reason == "no_signing" and dropped.n_frames > 45 and dropped.signing_ms < 100
    n_first = int(np.sum(raw.timestamps_ms < 3000))
    assert calls and max(calls) == n_first  # every call was for the first window
    assert len(calls) == 1 + engine.stats.partials_emitted

    # The same stretch on its own, with fixed-length boundaries: no call at all.
    calls.clear()
    tail = RawKeypoints(
        timestamps_ms=raw.timestamps_ms[raw.timestamps_ms >= 3200],
        keypoints=raw.keypoints[raw.timestamps_ms >= 3200],
        scores=raw.scores[raw.timestamps_ms >= 3200],
        person_found=raw.person_found[raw.timestamps_ms >= 3200],
        width=raw.width,
        height=raw.height,
    )
    cfg = StreamConfig(boundary=FixedLengthConfig(length_ms=1000), min_segment_ms=500)
    engine = StreamingEngine(cfg, raw.frame_size, 256)
    result = replay(engine, iter_raw(tail), offline_decoder(offline), pace="instant")
    assert calls == [] and engine.stats.finals_queued == 0
    assert engine.stats.segments_dropped_no_signing >= 1


def test_an_empty_scene_never_reaches_the_model(
    offline: OfflineTranslator, core: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real pose extraction on frames with nobody in them (black, and noise), streamed with
    the default settings. The model would answer such input with an invented sentence."""
    rng = np.random.default_rng(0)
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    noise = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    seen = [core.process(image, 0) for image in (black, noise)]
    calls = _spy(offline, monkeypatch)
    for cfg in (StreamConfig(), StreamConfig(boundary=FixedLengthConfig(length_ms=2000))):
        engine = StreamingEngine(cfg, (640, 480), 256)
        t = 0
        while t < 20_000:
            frame = seen[(t // 5000) % 2]
            assert engine.push(t, frame.keypoints, frame.scores, frame.person_found) is not None
            assert engine.next_job() is None
            t += int(rng.integers(28, 40))
        engine.flush()
        assert engine.next_job() is None
        s = engine.stats
        assert s.finals_queued == 0 and s.partials_requested == 0
        assert s.segments_dropped_no_signing >= 2
    assert calls == []


# --- a live-paced replay with the real model: commit rules, offline agreement, memory ----------


def test_live_paced_replay_commits_once_matches_offline_and_holds_bounded_state(
    offline: OfflineTranslator, cached_raw: Callable[[str], RawKeypoints]
) -> None:
    """Forty seconds of the continuous video at live pacing on the stream's own clock, with
    the default settings (fixed 8 s windows) and the real model decoding on CPU, where a
    decode can outlast the hop. Memory is sampled and printed; the hard assertions are on
    what the engine holds. The 30-minute real-model run is the user's to launch."""
    raw = cached_raw("continuous")
    cfg = StreamConfig()
    engine = StreamingEngine(cfg, raw.frame_size, offline.loaded.translator.cfg.max_frames)
    samples: list[dict[str, float]] = []

    def on_frame(t_ms: int) -> None:
        if t_ms >= 5000 * len(samples):
            samples.append({"stream_s": t_ms / 1000, **memory_snapshot(),
                            "buffered_frames": engine.buffered_frames})  # fmt: skip

    result = replay(
        engine, iter_raw(raw, loop_to_ms=40_000), offline_decoder(offline), on_frame=on_frame
    )
    print("\nlive-paced replay, 40 s, OpenASL on CPU:")
    print("  timing:", json.dumps(result.timing_summary()))
    print("  stats:", json.dumps(result.stats.to_dict()))
    for row in samples:
        print("  memory:", json.dumps(row))
    for e in result.finals:
        print(f"  FINAL {e.start_ms}-{e.end_ms}: {e.text!r}")

    _check_commit_rules(result.events)
    finals = result.finals
    assert len(finals) == 5 and finals[-1].closed_by == "flush"
    assert finals[0].start_ms == int(raw.timestamps_ms[0])
    for previous, e in itertools.pairwise(finals):
        assert e.start_ms == previous.end_ms  # fixed-length windows tile the stream
        assert 8000 <= previous.end_ms - previous.start_ms < 8050
    for e in finals:  # each committed sentence is the offline translation of its own slice
        expected = offline.translate_raw(raw, start_ms=e.start_ms, end_ms=e.end_ms)
        assert e.text == expected.text and e.n_frames == expected.n_frames <= 256
        assert e.latency_ms is not None and e.latency_ms > 0
    s = result.stats
    assert s.hops_due == s.hops_skipped_busy + s.hops_skipped_no_signing + s.partials_requested
    assert s.partials_emitted + s.partials_discarded_stale <= s.partials_requested
    assert s.max_buffered_frames <= 257 and s.max_pending_finals <= cfg.max_pending_finals
    assert s.finals_dropped_backlog == 0 and s.decode_failures == 0
    assert len(samples) >= 8 and all("rss_mb" in row for row in samples)


# --- the command ---------------------------------------------------------------------------------


def test_stream_replay_command(
    offline: OfflineTranslator,
    clip_path: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command end to end. It decodes with the real model, but reuses the instance that
    is already loaded instead of loading a second copy (the command's own load call is the
    one `s2e translate` uses, which has its own test). Keypoints come from the project's
    cache, as in normal use."""
    clip = clip_path("hands_visible")
    expected = offline.translate_file(clip).text
    capsys.readouterr()
    loads: list[Any] = []

    def already_loaded(paths: Any, model_cfg: Any, keypoint_cfg: Any, **kwargs: Any) -> Any:
        loads.append(kwargs)
        return offline.loaded

    monkeypatch.setattr(loader, "load_translator", already_loaded)
    monkeypatch.chdir(repo_root)

    assert main(["stream-replay", str(clip), "--device", "cpu", "--json", "--pace", "instant"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    finals = [r for r in rows if r["type"] == "final"]
    assert [r["text"] for r in finals] == [expected]
    assert [r["type"] for r in rows].count("partial") == 4
    summary = rows[-1]
    assert summary["type"] == "summary" and summary["model"]["decoding"] == "real"
    assert summary["model"]["device"] == "cpu" and summary["stats"]["finals_emitted"] == 1
    assert summary["stream_config"]["boundary"]["kind"] == "fixed_length"
    assert len(loads) == 1 and loads[0]["device"] == "cpu"

    # Decoding disabled: no model, no text, and the output says so.
    report = tmp_path / "report.json"
    args = ["stream-replay", str(clip), "--no-decode", "--assumed-decode-ms", "900"]
    assert (
        main([*args, "--length-ms", "2000", "--memory-every-s", "2", "--report", str(report)]) == 0
    )
    out = capsys.readouterr().out
    assert "(decoding disabled)" in out and expected not in out
    assert out.count("FINAL") == 2 and "memory" in out
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["model"] == {"decoding": "disabled"} and saved["timing"]["decoding"] == "disabled"
    assert len(loads) == 1  # --no-decode never asks for the model
    assert len(saved["memory"]) == 3 and saved["stats"]["segments_dropped_too_short"] == 1

    # Oracle boundaries from the command line, and bad input.
    assert main([*args, "--segments", "500-2500,3000-4900", "--hide-partials"]) == 0
    out = capsys.readouterr().out
    assert out.count("FINAL") == 2 and "partial" not in out
    assert main([*args, "--segments", "500"]) == 2
    assert "bad segment" in capsys.readouterr().err
    assert main(["stream-replay", str(tmp_path / "absent.mp4"), "--no-decode"]) == 1
    assert "not found" in capsys.readouterr().err
