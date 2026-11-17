"""The boundary interface, the per-frame signing claim, and the two simple detectors
(`models/boundary.py`). Pure logic; in the quick check. Track F's detector gets its own
tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from signtoenglish.models import boundary
from signtoenglish.models.boundary import (
    BoundaryError,
    FixedLength,
    FixedLengthConfig,
    Observation,
    OracleBoundaries,
    OracleConfig,
    SigningEvidenceConfig,
    build_detector,
    hand_in_view,
    parse_segments,
)
from signtoenglish.preprocess.schema import NUM_KEYPOINTS, PART_INDICES

SIZE = (640, 480)


def _obs(t: int, make_frame: Any, **kw: Any) -> Observation:
    keypoints, scores = make_frame(**kw)
    return Observation(t, keypoints, scores, True, SIZE)


def _events(detector: Any, times: list[int], make_frame: Any) -> list[tuple[int, str, int]]:
    out = []
    for t in times:
        for event in detector.update(_obs(t, make_frame)).events:
            out.append((t, event.kind, event.t_ms))
    return out


def test_hand_indices_are_the_base_models() -> None:
    """`models` may not import `preprocess`, so the ranges are repeated there."""
    assert tuple(range(*boundary.LEFT_HAND)) == PART_INDICES["left"]
    assert tuple(range(*boundary.RIGHT_HAND)) == PART_INDICES["right"]
    assert boundary.NUM_KEYPOINTS == NUM_KEYPOINTS


def test_hand_in_view_needs_a_confident_hand_inside_the_frame(make_frame: Any) -> None:
    cfg = SigningEvidenceConfig()
    assert hand_in_view(_obs(0, make_frame), cfg)
    assert not hand_in_view(_obs(0, make_frame, score=0.1), cfg)
    # Above the base model's 0.3 mask, below this rule: a lowered hand's typical score.
    assert not hand_in_view(_obs(0, make_frame, score=0.35), cfg)
    assert not hand_in_view(_obs(0, make_frame, score=0.49), cfg)
    assert hand_in_view(_obs(0, make_frame, score=0.5), cfg)

    one = _obs(0, make_frame, score=0.1)  # one hand is enough, either one
    one.scores[112:133] = 0.9
    assert hand_in_view(one, cfg)
    one.scores[112:133], one.scores[91:112] = 0.1, 0.9
    assert hand_in_view(one, cfg)

    out = _obs(0, make_frame)
    out.keypoints[91:133] = (700.0, 240.0)  # confident, but outside the frame
    assert not hand_in_view(out, cfg)
    edge = _obs(0, make_frame)
    edge.keypoints[91:133] = (320.0, 240.0)
    edge.keypoints[91:95] = (320.0, 480.0)  # 4 of 21 points off the bottom edge: 81% inside
    edge.keypoints[112:117] = (320.0, 481.0)  # 5 of 21: 76% inside
    edge.scores[91:112] = 0.1
    assert not hand_in_view(edge, cfg)
    edge.scores[91:112] = 0.9
    assert hand_in_view(edge, cfg)
    # Body and face scores play no part.
    body = _obs(0, make_frame, score=0.1)
    body.scores[:91] = 1.0
    assert not hand_in_view(body, cfg)
    with pytest.raises(Exception, match="greater than 0"):
        SigningEvidenceConfig(hand_score_min=0.0)


def test_fixed_length_cuts_by_time_and_restarts_after_a_gap_or_a_forced_cut(
    make_frame: Any,
) -> None:
    detector = FixedLength(FixedLengthConfig(length_ms=1000))
    assert _events(detector, [10, 400, 900, 1009, 1010, 1500, 2100], make_frame) == [
        (10, "start", 10),
        (1010, "end", 1010),
        (1010, "start", 1010),
        (2100, "end", 2100),
        (2100, "start", 2100),
    ]
    detector.notify_forced_cut(2500)  # the engine opened a segment at 2500 itself
    assert _events(detector, [2600, 3400, 3500], make_frame) == [
        (3500, "end", 3500),
        (3500, "start", 3500),
    ]
    detector.notify_gap(3500, 9000)  # the engine closed the segment; a new one must start
    assert _events(detector, [9000, 9900, 10000], make_frame) == [
        (9000, "start", 9000),
        (10000, "end", 10000),
        (10000, "start", 10000),
    ]
    assert detector.update(_obs(10100, make_frame)).signing
    assert not detector.update(_obs(10200, make_frame, hands=False)).signing
    assert detector.lookback_ms == 0


def test_oracle_reports_exactly_the_given_boundaries(make_frame: Any) -> None:
    cfg = OracleConfig(segments=((100, 300), (300, 500), (800, 900), (2000, 2100)))
    detector = OracleBoundaries(cfg)
    times = [0, 90, 100, 250, 310, 499, 500, 700, 850, 1500, 1600]
    assert _events(detector, times, make_frame) == [
        (100, "start", 100),
        (310, "end", 300),  # reported at the first frame past the end, with the given time
        (310, "start", 300),
        (500, "end", 500),
        (850, "start", 800),
        (1500, "end", 900),
    ]
    # A window with no frame in it is passed over without events.
    skipping = OracleBoundaries(OracleConfig(segments=((100, 200), (300, 400))))
    assert _events(skipping, [0, 250, 350, 450], make_frame) == [
        (350, "start", 300),
        (450, "end", 400),
    ]
    # After a gap inside a window, the rest of it starts at the first frame that arrived.
    gap = OracleBoundaries(OracleConfig(segments=((0, 1000),)))
    assert _events(gap, [0, 100], make_frame) == [(0, "start", 0)]
    gap.notify_gap(100, 700)
    assert _events(gap, [700, 1000], make_frame) == [(700, "start", 700), (1000, "end", 1000)]


def test_oracle_refuses_bad_segments() -> None:
    with pytest.raises(BoundaryError, match="empty or reversed"):
        OracleBoundaries(OracleConfig(segments=((500, 500),)))
    with pytest.raises(BoundaryError, match="without overlap"):
        OracleBoundaries(OracleConfig(segments=((0, 500), (400, 900))))
    with pytest.raises(BoundaryError, match="without overlap"):
        OracleBoundaries(OracleConfig(segments=((1000, 2000), (0, 500))))


def test_parse_segments_and_build_detector() -> None:
    assert parse_segments("2000-9000, 12000-20000") == ((2000, 9000), (12000, 20000))
    for bad in ("2000", "a-b", "1-2;3-4", ""):
        with pytest.raises(BoundaryError, match="bad segment"):
            parse_segments(bad)
    evidence = SigningEvidenceConfig(hand_score_min=0.7)
    fixed = build_detector(FixedLengthConfig(), evidence)
    assert isinstance(fixed, FixedLength) and fixed.evidence is evidence
    assert isinstance(build_detector(OracleConfig(segments=((0, 1),))), OracleBoundaries)
    with pytest.raises(BoundaryError, match="no boundary detector"):
        build_detector(object())
    assert np.isclose(FixedLengthConfig().length_ms, 8000)
