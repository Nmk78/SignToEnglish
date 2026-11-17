"""Scoring of a streamed transcript (`eval/stream_eval.py`): boundary F1, alignment of
committed sentences to marked segments, and the text metrics on them. Hand-computed
values; no model. The oracle check against the real model is in
tests/infer/test_streaming_model.py."""

from __future__ import annotations

import pytest

from signtoenglish.eval.stream_eval import (
    ReferenceSegment,
    StreamEvalError,
    align_committed,
    boundary_f1,
    score_stream,
    segments_from_row,
)
from signtoenglish.infer.streaming import StreamEvent


def _final(segment_id: int, start: int, end: int, text: str) -> StreamEvent:
    return StreamEvent("final", segment_id, text, start, end, n_frames=10, signing_ms=500)


def test_boundary_f1_matches_each_mark_at_most_once() -> None:
    score = boundary_f1([1000, 5200, 9000, 9400], [900, 5000, 12000])
    # 1000~900 and 5200~5000 match; 9000 and 9400 have no mark within a second; 12000 is missed.
    assert score.matches == ((1000, 900), (5200, 5000))
    assert (score.n_predicted, score.n_reference, score.n_matched) == (4, 3, 2)
    assert score.precision == pytest.approx(2 / 4) and score.recall == pytest.approx(2 / 3)
    assert score.f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))

    # Two predictions near one mark: only one counts.
    twice = boundary_f1([4800, 5300], [5000])
    assert (twice.n_matched, twice.precision, twice.recall) == (1, 0.5, 1.0)
    # A prediction that could take either mark leaves the other for its neighbour.
    chain = boundary_f1([1900, 2900], [1000, 2000])
    assert chain.matches == ((1900, 1000), (2900, 2000)) and chain.f1 == 1.0
    # The tolerance is inclusive and can be changed; order of the inputs does not matter.
    assert boundary_f1([2000], [1000]).n_matched == 1
    assert boundary_f1([2001], [1000]).n_matched == 0
    assert boundary_f1([2001], [1000], tolerance_ms=1500).n_matched == 1
    assert boundary_f1([9000, 1000], [5000, 900, 9100]).n_matched == 2
    # Nothing to get right is not a perfect score.
    assert boundary_f1([], []).f1 == 0.0
    assert boundary_f1([], [1000]).recall == 0.0 and boundary_f1([1000], []).precision == 0.0
    with pytest.raises(StreamEvalError):
        boundary_f1([1], [1], tolerance_ms=-1)


REFS = [
    ReferenceSegment("Hello, my name is Alex.", 0, 3000, "s001"),
    ReferenceSegment("Nice to meet you.", 3500, 6000, "s002"),
    ReferenceSegment("Where are you from?", 6500, 9000, "s003"),
]


def test_committed_sentences_go_to_the_marked_segment_they_overlap_most() -> None:
    events = [
        _final(0, 0, 2000, "Hello, my"),
        _final(1, 2000, 4000, "name is Alex."),  # 1000 ms in the first, 500 ms in the second
        _final(2, 4000, 6200, "Nice to meet you."),
        _final(4, 9500, 11000, "Something after the take."),  # overlaps nothing
        StreamEvent("partial", 5, "ignored", 6500, 8000, 10, 500),
        StreamEvent("dropped", 3, "", 6500, 9000, 10, 0, reason="no_signing"),
    ]
    aligned, extra = align_committed(events, REFS)
    assert [a.hypothesis for a in aligned] == ["Hello, my name is Alex.", "Nice to meet you.", ""]
    assert [a.segment_ids for a in aligned] == [(0, 1), (2,), ()]
    assert extra == 1

    score = score_stream(events, REFS)
    assert (score.n_missed, score.n_extra) == (1, 1)
    assert score.metrics.n == 3
    assert score.metrics.exact_match.n_match == 2
    assert score.metrics.exact_match.rate == pytest.approx(100 * 2 / 3)
    assert score.metrics.bleu.signature.startswith("nrefs:1|case:mixed")
    # Committed ends 2000, 4000, 6200, 11000 against marks 3000, 6000, 9000: 2000~3000 and
    # 6200~6000 match; 4000 falls between (its mark is taken or too far), 11000 is too late.
    assert score.boundary.matches == ((2000, 3000), (6200, 6000))
    assert score.boundary.f1 == pytest.approx(2 * (2 / 4) * (2 / 3) / (2 / 4 + 2 / 3))
    report = score.to_dict()
    assert report["sentences"][2] == {
        "reference": "Where are you from?",
        "hypothesis": "",
        "start_ms": 6500,
        "end_ms": 9000,
        "segment_ids": [],
    }
    assert report["metrics"]["bleu"]["signature"] == score.metrics.bleu.signature


def test_perfect_stream_scores_perfectly() -> None:
    events = [_final(i, r.start_ms, r.end_ms, r.text) for i, r in enumerate(REFS)]
    score = score_stream(events, REFS)
    assert score.metrics.exact_match.rate == 100.0 and score.metrics.bleu.bleu4 == pytest.approx(
        100.0
    )
    assert score.boundary.f1 == 1.0 and (score.n_missed, score.n_extra) == (0, 0)
    with pytest.raises(StreamEvalError, match="no marked segments"):
        score_stream(events, [])


def test_segments_from_a_continuous_take_row() -> None:
    row = {
        "id": "take-1",
        "segments": [
            {"sentence_id": "s002", "text": "Nice to meet you.", "start_ms": 3500, "end_ms": 6000},
            {"sentence_id": "s001", "text": "Hello.", "start_ms": 0, "end_ms": 3000},
        ],
    }
    refs = segments_from_row(row)
    assert [(r.sentence_id, r.start_ms, r.end_ms) for r in refs] == [
        ("s001", 0, 3000),
        ("s002", 3500, 6000),
    ]
    for bad in (
        {},
        {"segments": []},
        {"segments": [{"text": "x", "start_ms": 0}]},
        {"segments": [{"text": "x", "start_ms": 5, "end_ms": 5}]},
    ):
        with pytest.raises(StreamEvalError):
            segments_from_row(bad)
