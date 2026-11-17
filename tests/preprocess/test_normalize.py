"""`normalize_keypoints`: the base model's rule, tested on small arrays (no model files).

Parity with the reference implementation on real clips is in `test_parity.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from signtoenglish.preprocess.normalize import normalize_keypoints, part_present
from signtoenglish.preprocess.schema import (
    NUM_KEYPOINTS,
    PART_INDICES,
    PART_REFERENCE,
    PARTS,
    NormalizeConfig,
)

CFG = NormalizeConfig()
SIZE = (1280, 720)
REFERENCE_POINTS = (91, 112, 53)  # left-hand root, right-hand root, face reference


def make(
    t: int = 12, seed: int = 0, score: float = 0.9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    kps = rng.uniform([200, 100], [1000, 600], size=(t, NUM_KEYPOINTS, 2)).astype(np.float32)
    scores = np.full((t, NUM_KEYPOINTS), score, dtype=np.float32)
    return kps, scores, np.ones(t, dtype=np.bool_)


def run(kps: np.ndarray, scores: np.ndarray, found: np.ndarray):
    return normalize_keypoints(kps, scores, found, SIZE, CFG)


def test_group_layout_matches_the_base_model() -> None:
    assert PARTS == ("body", "left", "right", "face_all")
    assert [len(PART_INDICES[p]) for p in PARTS] == [9, 21, 21, 18]
    assert PART_INDICES["body"] == (0, 3, 4, 5, 6, 7, 8, 9, 10)
    assert PART_INDICES["left"][0] == 91 and PART_INDICES["right"][0] == 112
    assert PART_INDICES["face_all"] == (
        23,
        25,
        27,
        29,
        31,
        33,
        35,
        37,
        39,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        53,
    )
    refs = {PART_INDICES[p][PART_REFERENCE[p]] for p in PARTS if PART_REFERENCE[p] is not None}
    assert refs == set(REFERENCE_POINTS)


def test_shapes_dtypes_and_ranges() -> None:
    inputs, mask = run(*make())
    assert list(inputs) == list(PARTS) == list(mask)
    for part in PARTS:
        v = len(PART_INDICES[part])
        assert inputs[part].shape == (12, v, 3) and inputs[part].dtype == np.float32
        assert mask[part].shape == (12, v) and mask[part].dtype == np.bool_
        assert inputs[part].min() >= -1 and inputs[part].max() <= 1
        assert mask[part].all()


def test_body_box_is_computed_over_the_whole_clip_by_hand() -> None:
    kps, scores, found = make(t=2)
    body = list(PART_INDICES["body"])
    # frame 0: every body point at (320, 180); frame 1: at (960, 540). In [W, H] units that is
    # (0.25, 0.25) and (0.75, 0.75): a box of side 0.5 with corner (0.25, 0.25).
    kps[0, body] = (320, 180)
    kps[1, body] = (960, 540)
    inputs, _ = run(kps, scores, found)
    np.testing.assert_allclose(inputs["body"][0, :, :2], -1.0, atol=1e-6)
    np.testing.assert_allclose(inputs["body"][1, :, :2], 1.0, atol=1e-6)
    np.testing.assert_allclose(inputs["body"][..., 2], 0.9, atol=1e-6)
    # A hand point 64 px right and 36 px below its root is (0.05, 0.05) in [W, H] units,
    # divided by the body box side 0.5 gives (0.1, 0.1).
    kps[:, 92] = kps[:, 91] + np.array([64, 36], dtype=np.float32)
    inputs, _ = run(kps, scores, found)
    np.testing.assert_allclose(inputs["left"][:, 1, :2], 0.1, atol=1e-5)
    np.testing.assert_allclose(inputs["left"][:, 0, :2], 0.0, atol=1e-7)  # the root itself


def test_low_score_keypoints_become_zero_and_are_masked() -> None:
    kps, scores, found = make()
    scores[3, 95] = 0.25
    scores[4, 5] = 0.1  # a body point
    inputs, mask = run(kps, scores, found)
    assert not mask["left"][3, 4] and (inputs["left"][3, 4] == 0).all()
    assert not mask["body"][4, 3] and (inputs["body"][4, 3] == 0).all()
    assert mask["left"].sum() == mask["left"].size - 1


def test_threshold_is_compared_as_in_training() -> None:
    # float32(0.3) is 0.30000001192...: the reference widened scores to float64 before
    # comparing with 0.3, so that value counts as above the threshold.
    kps, scores, found = make()
    scores[0, 100] = np.float32(0.3)
    scores[0, 101] = np.nextafter(np.float32(0.3), np.float32(0))
    _, mask = run(kps, scores, found)
    assert mask["left"][0, 9] and not mask["left"][0, 10]


def test_scores_above_one_are_clipped_like_the_reference() -> None:
    kps, scores, found = make()
    scores[:, 112:133] = 1.08  # the lightweight pose model does return values just over 1
    inputs, _ = run(kps, scores, found)
    assert inputs["right"][..., 2].max() == 1.0


def test_changing_a_masked_keypoint_does_not_change_the_output() -> None:
    """Holds for every used keypoint except the three reference points (next test)."""
    kps, scores, found = make(seed=1)
    used = sorted({i for p in PARTS for i in PART_INDICES[p]} - set(REFERENCE_POINTS))
    rng = np.random.default_rng(2)
    masked = [(int(rng.integers(12)), i) for i in used]
    for t, i in masked:
        scores[t, i] = 0.05
    before, mask_before = run(kps, scores, found)
    moved = kps.copy()
    for t, i in masked:
        moved[t, i] = (-5000.0, 99999.0)
    after, mask_after = run(moved, scores, found)
    for part in PARTS:
        np.testing.assert_array_equal(before[part], after[part])
        np.testing.assert_array_equal(mask_before[part], mask_after[part])


def test_unused_keypoints_never_matter() -> None:
    kps, scores, found = make(seed=3)
    used = {i for p in PARTS for i in PART_INDICES[p]}
    unused = [i for i in range(NUM_KEYPOINTS) if i not in used]
    assert len(used) == 69
    before, _ = run(kps, scores, found)
    kps[:, unused] = 12345.0
    scores[:, unused] = 0.0
    after, _ = run(kps, scores, found)
    for part in PARTS:
        np.testing.assert_array_equal(before[part], after[part])


@pytest.mark.parametrize(("point", "part"), [(91, "left"), (112, "right"), (53, "face_all")])
def test_a_masked_reference_point_still_moves_its_group(point: int, part: str) -> None:
    """Base-model behaviour, kept for parity (`datasets.py:36,40,44` with `:71`): a hand is
    measured from its root, and the face from point 53, using that point's raw position
    even when its score is below the threshold. The point itself is zeroed; the rest of its
    group shifts. PLAN.md's wording for this test does not hold for these three points."""
    kps, scores, found = make(seed=4)
    scores[:, point] = 0.05
    before, mask = run(kps, scores, found)
    ref = PART_REFERENCE[part]
    assert ref is not None and not mask[part][:, ref].any()
    assert (before[part][:, ref] == 0).all()
    moved = kps.copy()
    moved[:, point] += np.array([40.0, -25.0], dtype=np.float32)
    after, _ = run(moved, scores, found)
    assert not np.array_equal(before[part], after[part])
    for other in PARTS:
        if other != part:
            np.testing.assert_array_equal(before[other], after[other])


def test_no_usable_body_zeroes_everything() -> None:
    kps, scores, found = make()
    scores[:, list(PART_INDICES["body"])] = 0.1
    scores[0, [0, 3, 4]] = 0.9  # three valid body observations: one short of four
    inputs, mask = run(kps, scores, found)
    for part in PARTS:
        assert not inputs[part].any() and not mask[part].any()
    scores[0, 5] = 0.9  # the fourth
    inputs, mask = run(kps, scores, found)
    assert inputs["left"].any() and mask["left"].all()


def test_all_low_scores_give_all_false_masks() -> None:
    inputs, mask = run(*make(score=0.2))
    for part in PARTS:
        assert not inputs[part].any() and not mask[part].any()
    assert not any(present.any() for present in part_present(mask).values())


def test_person_found_is_not_used() -> None:
    kps, scores, found = make(seed=5)
    a, _ = run(kps, scores, found)
    b, _ = run(kps, scores, np.zeros_like(found))
    for part in PARTS:
        np.testing.assert_array_equal(a[part], b[part])


def test_pure_and_deterministic() -> None:
    kps, scores, found = make(seed=6)
    scores[2:5, 91:112] = 0.1
    copies = (kps.copy(), scores.copy(), found.copy())
    a, mask_a = run(kps, scores, found)
    b, mask_b = run(kps, scores, found)
    for original, copy in zip((kps, scores, found), copies, strict=True):
        np.testing.assert_array_equal(original, copy)
    for part in PARTS:
        assert a[part].tobytes() == b[part].tobytes()
        np.testing.assert_array_equal(mask_a[part], mask_b[part])


def test_works_on_a_slice_and_uses_that_slices_own_body_box() -> None:
    kps, scores, found = make(t=20, seed=7)
    kps[10:, list(PART_INDICES["body"])] *= 0.5  # the second half sits elsewhere
    whole, _ = run(kps, scores, found)
    part_, _ = run(kps[:10], scores[:10], found[:10])
    assert part_["body"].shape[0] == 10
    # Whole-clip statistics: the same frames normalize differently inside a longer segment.
    assert not np.allclose(whole["body"][:10], part_["body"])
    again, _ = run(kps[:10].copy(), scores[:10].copy(), found[:10].copy())
    np.testing.assert_array_equal(part_["body"], again["body"])


def test_empty_input() -> None:
    inputs, mask = normalize_keypoints(
        np.zeros((0, NUM_KEYPOINTS, 2), np.float32),
        np.zeros((0, NUM_KEYPOINTS), np.float32),
        np.zeros((0,), np.bool_),
        SIZE,
        CFG,
    )
    assert all(inputs[p].shape == (0, len(PART_INDICES[p]), 3) for p in PARTS)
    assert all(mask[p].shape == (0, len(PART_INDICES[p])) for p in PARTS)


def test_bad_shapes_are_rejected() -> None:
    kps, scores, found = make()
    with pytest.raises(ValueError, match="expected keypoints"):
        normalize_keypoints(kps[:, :100], scores, found, SIZE, CFG)
    with pytest.raises(ValueError, match="person_found"):
        normalize_keypoints(kps, scores, found[:3], SIZE, CFG)
    with pytest.raises(ValueError, match="frame size"):
        normalize_keypoints(kps, scores, found, (0, 720), CFG)


def test_there_is_exactly_one_normalization_in_src(repo_root: Path) -> None:
    """`normalize_keypoints` is defined once, and nothing else in src/ looks like a second
    normalization: no other function with "normaliz" in its name, and no other module that
    applies the keypoint score threshold from `NormalizeConfig`."""
    src = repo_root / "src"
    home = src / "signtoenglish" / "preprocess" / "normalize.py"
    definitions: list[str] = []
    lookalikes: list[str] = []
    threshold_users: list[str] = []
    for path in sorted(src.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name == "normalize_keypoints":
                    definitions.append(str(path.relative_to(src)))
                elif "normaliz" in node.name.lower() and (
                    "keypoint" in node.name.lower()
                    or (path.parent.name == "preprocess" and path != home)
                ):
                    lookalikes.append(f"{path.relative_to(src)}:{node.name}")
            if isinstance(node, ast.Attribute) and node.attr == "score_threshold" and path != home:
                threshold_users.append(f"{path.relative_to(src)}:{node.lineno}")
    assert definitions == ["signtoenglish/preprocess/normalize.py"]
    assert lookalikes == []
    assert threshold_users == []
