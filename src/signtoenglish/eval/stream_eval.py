"""Scoring a streamed transcript (ARCHITECTURE.md, section 7, "Streaming").

Three things, all computed from the engine's committed sentences and their times:

- `boundary_f1`: predicted sentence ends against marked ones, within a tolerance
  (default one second), each mark matched at most once.
- `score_stream`: the text metrics of `eval/metrics.py` on committed sentences aligned to
  the marked segments of a continuous take. Each committed sentence goes to the marked
  segment it overlaps most in time; a marked segment's hypothesis is its committed
  sentences joined in time order, or the empty string if it got none.
- `oracle_check`: the Step 3.1 acceptance check. The same windows are translated offline
  (`OfflineTranslator.translate_raw`) and by the streaming engine driven by oracle
  boundaries, and the two texts are compared window by window.

Marked segments are plain `ReferenceSegment`s. `segments_from_row` reads them from a
continuous-take manifest row in the format of ARCHITECTURE.md section 4.2; validating
manifests is `data.manifest`'s job, not this module's.

Run the oracle check from the command line (it loads the model):

    uv run --frozen python -m signtoenglish.eval.stream_eval VIDEO --segments 2000-9000,12000-19000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from signtoenglish.eval.metrics import TextMetrics, compute_metrics
from signtoenglish.infer.offline import OfflineTranslator
from signtoenglish.infer.streaming import (
    RawKeypoints,
    ReplayResult,
    StreamConfig,
    StreamEvent,
    StreamingEngine,
    iter_raw,
    offline_decoder,
    replay,
)
from signtoenglish.models.boundary import OracleConfig

DEFAULT_TOLERANCE_MS = 1000


class StreamEvalError(ValueError):
    """The inputs cannot be scored; the message says why."""


@dataclass(frozen=True, slots=True)
class ReferenceSegment:
    """One marked sentence of a continuous take: `[start_ms, end_ms)` and its English."""

    text: str
    start_ms: int
    end_ms: int
    sentence_id: str | None = None


def segments_from_row(row: Mapping[str, Any]) -> list[ReferenceSegment]:
    """The `segments` list of a continuous-take manifest row, in time order."""
    segments = row.get("segments")
    if not isinstance(segments, list) or not segments:
        raise StreamEvalError("continuous-take row has no `segments` list")
    out = []
    for i, seg in enumerate(segments):
        try:
            ref = ReferenceSegment(
                text=str(seg["text"]),
                start_ms=int(seg["start_ms"]),
                end_ms=int(seg["end_ms"]),
                sentence_id=seg.get("sentence_id"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StreamEvalError(f"segment {i}: needs text, start_ms, end_ms ({exc})") from exc
        if ref.end_ms <= ref.start_ms:
            raise StreamEvalError(
                f"segment {i}: end {ref.end_ms} is not after start {ref.start_ms}"
            )
        out.append(ref)
    return sorted(out, key=lambda r: r.start_ms)


# --- boundaries ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundaryScore:
    tolerance_ms: int
    n_predicted: int
    n_reference: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    #: (predicted, reference) times of each match.
    matches: tuple[tuple[int, int], ...]


def boundary_f1(
    predicted_ms: Sequence[int],
    reference_ms: Sequence[int],
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
) -> BoundaryScore:
    """Precision, recall, and F1 of predicted boundary times against marked ones.

    A prediction matches a mark when they are at most `tolerance_ms` apart; every
    prediction and every mark is used at most once. Both lists are walked in time order and
    the earlier unmatched time is given up first, which finds the largest possible set of
    matches when one tolerance applies to all. With nothing predicted and nothing marked
    the scores are 0, not 1: there was nothing to get right.
    """
    if tolerance_ms < 0:
        raise StreamEvalError(f"tolerance must not be negative, got {tolerance_ms}")
    predicted, reference = sorted(predicted_ms), sorted(reference_ms)
    matches: list[tuple[int, int]] = []
    i = j = 0
    while i < len(predicted) and j < len(reference):
        if abs(predicted[i] - reference[j]) <= tolerance_ms:
            matches.append((predicted[i], reference[j]))
            i += 1
            j += 1
        elif predicted[i] < reference[j]:
            i += 1
        else:
            j += 1
    n = len(matches)
    precision = n / len(predicted) if predicted else 0.0
    recall = n / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if n else 0.0
    return BoundaryScore(
        tolerance_ms=tolerance_ms,
        n_predicted=len(predicted),
        n_reference=len(reference),
        n_matched=n,
        precision=precision,
        recall=recall,
        f1=f1,
        matches=tuple(matches),
    )


# --- text on aligned segments ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AlignedSentence:
    reference: ReferenceSegment
    hypothesis: str
    segment_ids: tuple[int, ...]  # the committed segments joined into the hypothesis


@dataclass(frozen=True, slots=True)
class StreamScore:
    metrics: TextMetrics
    boundary: BoundaryScore
    aligned: tuple[AlignedSentence, ...]
    n_missed: int  # marked sentences that got no committed text
    #: Committed sentences overlapping no marked sentence. They are in no hypothesis, so
    #: the text metrics do not see them; this count is how they are reported.
    n_extra: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "boundary": asdict(self.boundary),
            "n_missed": self.n_missed,
            "n_extra": self.n_extra,
            "sentences": [
                {
                    "reference": a.reference.text,
                    "hypothesis": a.hypothesis,
                    "start_ms": a.reference.start_ms,
                    "end_ms": a.reference.end_ms,
                    "segment_ids": list(a.segment_ids),
                }
                for a in self.aligned
            ],
        }


def _finals(events: Sequence[StreamEvent]) -> list[StreamEvent]:
    return sorted((e for e in events if e.type == "final"), key=lambda e: e.start_ms)


def align_committed(
    events: Sequence[StreamEvent], references: Sequence[ReferenceSegment]
) -> tuple[list[AlignedSentence], int]:
    """Give each committed sentence to the marked segment it overlaps most (the earlier
    one on a tie). Returns the aligned sentences in reference order and the number of
    committed sentences that overlap no marked segment."""
    refs = sorted(references, key=lambda r: r.start_ms)
    assigned: list[list[StreamEvent]] = [[] for _ in refs]
    extra = 0
    for event in _finals(events):
        best, best_overlap = -1, 0
        for i, ref in enumerate(refs):
            overlap = min(event.end_ms, ref.end_ms) - max(event.start_ms, ref.start_ms)
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        if best < 0:
            extra += 1
        else:
            assigned[best].append(event)
    aligned = [
        AlignedSentence(
            reference=ref,
            hypothesis=" ".join(e.text.strip() for e in got if e.text.strip()),
            segment_ids=tuple(e.segment_id for e in got),
        )
        for ref, got in zip(refs, assigned, strict=True)
    ]
    return aligned, extra


def score_stream(
    events: Sequence[StreamEvent],
    references: Sequence[ReferenceSegment],
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
) -> StreamScore:
    """Text metrics on aligned sentences and boundary F1 of the committed segments' end
    times against the marked end times."""
    if not references:
        raise StreamEvalError("no marked segments to score against")
    aligned, extra = align_committed(events, references)
    metrics = compute_metrics([a.hypothesis for a in aligned], [a.reference.text for a in aligned])
    boundary = boundary_f1(
        [e.end_ms for e in _finals(events)], [r.end_ms for r in references], tolerance_ms
    )
    return StreamScore(
        metrics=metrics,
        boundary=boundary,
        aligned=tuple(aligned),
        n_missed=sum(1 for a in aligned if not a.segment_ids),
        n_extra=extra,
    )


# --- streaming against offline, on the same windows ----------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowComparison:
    start_ms: int
    end_ms: int
    offline_text: str
    stream_text: str | None  # None: the engine committed nothing for this window
    stream_note: str  # why not, when it did not
    n_frames: int

    @property
    def equal(self) -> bool:
        return self.stream_text is not None and self.stream_text == self.offline_text


@dataclass(frozen=True, slots=True)
class OracleCheck:
    windows: tuple[WindowComparison, ...]
    replay: ReplayResult

    @property
    def all_equal(self) -> bool:
        return bool(self.windows) and all(w.equal for w in self.windows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_equal": self.all_equal,
            "windows": [{**asdict(w), "equal": w.equal} for w in self.windows],
            "stats": self.replay.stats.to_dict(),
        }


def oracle_check(
    offline: OfflineTranslator,
    raw: RawKeypoints,
    segments: Sequence[tuple[int, int]],
    cfg: StreamConfig | None = None,
) -> OracleCheck:
    """Translate each window offline, then replay the whole recording through the engine
    with those windows as oracle boundaries, and line the two up.

    Windows must fit the engine's limits (the model's maximum frame count, the maximum
    segment length); one that does not is cut by the engine and reported as unequal.
    """
    cfg = (cfg or StreamConfig()).model_copy(
        update={"boundary": OracleConfig(segments=tuple((int(a), int(b)) for a, b in segments))}
    )
    offline_results = [
        offline.translate_raw(raw, start_ms=a, end_ms=b, clip_id=f"window-{a}-{b}")
        for a, b in segments
    ]
    engine = StreamingEngine(cfg, raw.frame_size, offline.loaded.translator.cfg.max_frames)
    result = replay(engine, iter_raw(raw), offline_decoder(offline), pace="instant")
    by_start: dict[int, list[StreamEvent]] = {}
    for event in result.events:
        if event.type in ("final", "dropped", "error"):
            by_start.setdefault(event.start_ms, []).append(event)
    windows = []
    for (a, b), off in zip(segments, offline_results, strict=True):
        got = by_start.get(a, [])
        exact = [e for e in got if e.type == "final" and e.end_ms == b]
        if exact:
            text: str | None = exact[0].text
            note = ""
        else:
            text = None
            note = (
                "; ".join(
                    f"{e.type} {e.start_ms}..{e.end_ms} ms"
                    f" ({e.reason or 'closed by ' + str(e.closed_by)})"
                    for e in got
                )
                or "the engine opened no segment here"
            )
        windows.append(WindowComparison(a, b, off.text, text, note, off.n_frames))
    return OracleCheck(windows=tuple(windows), replay=result)


def main(argv: Sequence[str] | None = None) -> int:
    from signtoenglish.common.config import find_repo_root, load_paths
    from signtoenglish.infer.loader import load_keypoint_config, load_model_config, load_translator
    from signtoenglish.infer.streaming import StreamError, keypoints_for_replay, load_stream_config
    from signtoenglish.models.boundary import BoundaryError, parse_segments

    parser = argparse.ArgumentParser(
        prog="python -m signtoenglish.eval.stream_eval",
        description="Oracle check: streaming with given boundaries against offline translation "
        "of the same windows.",
    )
    parser.add_argument("video")
    parser.add_argument("--segments", required=True, help="START-END,START-END in milliseconds")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"])
    parser.add_argument("--stream-config")
    args = parser.parse_args(argv)
    try:
        segments = parse_segments(args.segments)
    except BoundaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    paths = load_paths(find_repo_root())
    keypoint_cfg = load_keypoint_config()
    try:
        raw, _ = keypoints_for_replay(Path(args.video), paths, keypoint_cfg)
    except (StreamError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    loaded = load_translator(
        paths,
        load_model_config(),
        keypoint_cfg,
        base_checkpoint=args.checkpoint,
        device=args.device,
    )
    check = oracle_check(
        OfflineTranslator(loaded, paths),
        raw,
        segments,
        load_stream_config(args.stream_config),
    )
    print(json.dumps(check.to_dict(), ensure_ascii=False, indent=2))
    return 0 if check.all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
