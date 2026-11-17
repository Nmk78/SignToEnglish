"""Extraction: the rate rule on plain timestamps, and the real pose estimator on real video.

Tests that take the `core`, `raw_of` or `clip_path` fixtures run RTMPose on the fixture
clips; see `conftest.py` for what happens where those files are absent.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from signtoenglish.common.config import PathsConfig
from signtoenglish.common.hashing import HashMismatchError
from signtoenglish.preprocess.extract import (
    LiveExtractor,
    PoseCore,
    RateReducer,
    extract_file,
    iter_keypoints,
)
from signtoenglish.preprocess.normalize import normalize_keypoints, part_present
from signtoenglish.preprocess.schema import (
    NUM_KEYPOINTS,
    KeypointConfig,
    NormalizeConfig,
    RawKeypoints,
)
from signtoenglish.preprocess.video_io import read_frames

NORM = NormalizeConfig()
#: A hand "is detected" in a frame when at least this many of its 21 points pass the base
#: model's score rule. This is a reporting threshold for the tests, not a model rule.
HAND_POINTS = 11


def kept(times: list[int], fps: float | None, slack: float = 0.25) -> list[int]:
    reducer = RateReducer(fps, slack)
    return [t for t in times if reducer.keep(t)]


# --- the rate rule (no model files needed) ---------------------------------------------


def test_source_at_the_model_rate_keeps_every_frame() -> None:
    times = [round(i * 1000 / 30) for i in range(300)]
    assert kept(times, 30.0) == times


def test_ntsc_and_slower_sources_keep_every_frame() -> None:
    ntsc = [round(i * 1001 / 30) for i in range(600)]  # 29.97 frames per second
    assert kept(ntsc, 30.0) == ntsc
    pal = [i * 40 for i in range(200)]  # 25 frames per second
    assert kept(pal, 30.0) == pal


def test_double_rate_source_keeps_every_second_frame() -> None:
    times = [round(i * 1000 / 60) for i in range(600)]
    out = kept(times, 30.0)
    assert out == times[::2]


def test_jittery_camera_timestamps_are_not_thinned() -> None:
    rng = np.random.default_rng(0)
    times = sorted({round(i * 1000 / 30 + float(rng.uniform(-6, 6))) for i in range(600)})
    out = kept(times, 30.0)
    assert len(out) >= 0.98 * len(times)


def test_no_model_rate_keeps_everything() -> None:
    times = [0, 3, 9, 10, 500]
    assert kept(times, None) == times


def test_duplicates_and_out_of_order_timestamps_are_rejected_and_counted() -> None:
    reducer = RateReducer(30.0)
    decisions = [reducer.keep(t) for t in [0, 33, 33, 20, 67, 67, 100]]
    assert decisions == [True, True, False, False, True, False, True]
    assert reducer.rejected_non_increasing == 3 and reducer.kept == 4 and reducer.seen == 7


def test_gap_in_the_stream_resumes_cleanly() -> None:
    times = [0, 33, 67, 5000, 5033, 5067]
    assert kept(times, 30.0) == times


def test_bad_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="model_rate_fps"):
        RateReducer(0)
    with pytest.raises(ValueError, match="rate_slack"):
        RateReducer(30, 1.0)


# --- model files are pinned ------------------------------------------------------------


def test_missing_model_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="never downloads"):
        PoseCore(KeypointConfig(), tmp_path)


def test_model_file_with_the_wrong_hash_is_refused(
    tmp_path: Path,
    keypoint_cfg: KeypointConfig,
    paths: PathsConfig,
    need_file: Callable[[Path, str], Path],
) -> None:
    source = need_file(
        paths.assets_root / keypoint_cfg.detector.relative_path, "detector model file"
    )
    target = tmp_path / keypoint_cfg.detector.relative_path
    target.parent.mkdir(parents=True)
    shutil.copyfile(source, target)
    with target.open("r+b") as fh:  # flip one byte
        fh.seek(1000)
        byte = fh.read(1)
        fh.seek(1000)
        fh.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(HashMismatchError):
        PoseCore(keypoint_cfg, tmp_path)


# --- real extraction on the fixture clips ----------------------------------------------


@pytest.mark.parametrize("stem", ["hands_visible", "hands_leave"])
def test_extraction_output_is_raw_and_well_formed(
    stem: str, raw_of: Callable[[str], RawKeypoints]
) -> None:
    raw = raw_of(stem)
    assert len(raw) == 150
    assert raw.keypoints.shape == (150, NUM_KEYPOINTS, 2) and raw.keypoints.dtype == np.float32
    assert raw.scores.shape == (150, NUM_KEYPOINTS) and raw.scores.dtype == np.float32
    assert raw.timestamps_ms.dtype == np.int64 and np.all(np.diff(raw.timestamps_ms) > 0)
    assert raw.frame_size == (1280, 720)
    assert raw.person_found.all()
    # Pixel coordinates, not normalized ones: the nose is somewhere inside a 1280x720 frame.
    assert 100 < float(raw.keypoints[:, 0, 0].mean()) < 1180
    assert np.isfinite(raw.keypoints).all() and np.isfinite(raw.scores).all()


@pytest.mark.parametrize("stem", ["hands_visible", "hands_leave"])
def test_file_path_and_live_path_give_identical_arrays(
    stem: str,
    core: PoseCore,
    raw_of: Callable[[str], RawKeypoints],
    clip_path: Callable[[str], Path],
) -> None:
    from_file = raw_of(stem)
    live = LiveExtractor(core)
    pushed = [live.push(frame.image, frame.t_ms) for frame in read_frames(clip_path(stem))]
    results = [p for p in pushed if p is not None]
    from_live = RawKeypoints.from_frames(results, *from_file.frame_size)
    assert from_live.timestamps_ms.tobytes() == from_file.timestamps_ms.tobytes()
    assert from_live.keypoints.tobytes() == from_file.keypoints.tobytes()
    assert from_live.scores.tobytes() == from_file.scores.tobytes()
    assert from_live.person_found.tobytes() == from_file.person_found.tobytes()


def test_two_runs_give_byte_identical_output(
    keypoint_cfg: KeypointConfig, paths: PathsConfig, raw_of: Callable[[str], RawKeypoints],
    clip_path: Callable[[str], Path],
) -> None:  # fmt: skip
    first = raw_of("hands_leave")
    second = extract_file(clip_path("hands_leave"), PoseCore(keypoint_cfg, paths.assets_root))
    for name in ("timestamps_ms", "keypoints", "scores", "person_found"):
        assert getattr(first, name).tobytes() == getattr(second, name).tobytes(), name
    a, _ = normalize_keypoints(
        first.keypoints, first.scores, first.person_found, first.frame_size, NORM
    )
    b, _ = normalize_keypoints(
        second.keypoints, second.scores, second.person_found, second.frame_size, NORM
    )
    assert all(a[p].tobytes() == b[p].tobytes() for p in a)


def test_rate_reduction_happens_before_pose_estimation_and_matches_live(
    tmp_path: Path,
    core: PoseCore,
    clip_path: Callable[[str], Path],
    video_writer: Callable[..., Path],
) -> None:
    """A 60 frames-per-second file: both paths run the estimator on the same half of the
    frames, chosen by timestamp."""
    source = [
        f.image[::4, ::4].copy()
        for f, _ in zip(read_frames(clip_path("hands_visible")), range(24), strict=False)
    ]
    doubled = [img for img in source for _ in range(2)]
    times = [round(i * 1000 / 60) for i in range(len(doubled))]
    path = video_writer(tmp_path / "sixty.mkv", doubled, times)

    calls: list[int] = []
    original = core.process

    def counting(image: np.ndarray, t_ms: int):  # type: ignore[no-untyped-def]
        calls.append(t_ms)
        return original(image, t_ms)

    core.process = counting  # type: ignore[method-assign]
    try:
        from_file = extract_file(path, core)
        file_calls, calls[:] = list(calls), []
        live = LiveExtractor(core)
        from_live = [r for f in read_frames(path) if (r := live.push(f.image, f.t_ms)) is not None]
        live_calls = list(calls)
    finally:
        core.process = original  # type: ignore[method-assign]

    assert file_calls == times[::2] == live_calls  # the estimator never saw the dropped frames
    assert from_file.timestamps_ms.tolist() == times[::2]
    assert np.stack([r.keypoints for r in from_live]).tobytes() == from_file.keypoints.tobytes()
    assert live.reducer.seen == 48 and live.reducer.kept == 24


def test_hands_are_detected_in_at_least_90_percent_of_hands_visible(
    raw_of: Callable[[str], RawKeypoints],
) -> None:
    raw = raw_of("hands_visible")
    _, mask = normalize_keypoints(raw.keypoints, raw.scores, raw.person_found, raw.frame_size, NORM)
    present = part_present(mask, HAND_POINTS)
    both = present["left"] & present["right"]
    rate = float(both.mean())
    print(
        f"hands_visible: both hands detected in {int(both.sum())}/{len(both)} frames ({rate:.1%})"
    )
    assert rate >= 0.90, f"both hands detected in only {rate:.1%} of frames"


def test_hands_leave_follows_the_base_models_score_rule(
    raw_of: Callable[[str], RawKeypoints],
) -> None:
    """What the base model's rule (score <= 0.3 becomes zeros) does on the clip where both
    hands are lowered out of the frame after about 3 s. It is not the clean "mask is false"
    that PLAN.md asks for, and this test pins the actual behaviour:

    - While the hands are in view, both pass in every frame.
    - Once lowered, the LEFT hand is masked in nearly every frame.
    - The lowered RIGHT hand is NOT reliably masked: the estimator still scores it just above
      0.3 and places it inside the frame, so its guessed points reach the model. This is the
      reference's behaviour; whether to add a stricter rule is an open decision
      (`docs/base_model_notes.md`). If the last assertion starts failing because the right
      hand is now masked, extraction changed: find out why before touching this test.
    """
    raw = raw_of("hands_leave")
    _, mask = normalize_keypoints(raw.keypoints, raw.scores, raw.person_found, raw.frame_size, NORM)
    present = part_present(mask, HAND_POINTS)
    in_view, lowered = slice(0, 85), slice(100, 150)
    assert present["left"][in_view].all() and present["right"][in_view].all()
    assert float(present["left"][lowered].mean()) <= 0.05
    assert float(mask["left"][lowered].mean()) <= 0.05  # point by point, too
    right_scores = raw.scores[lowered, 112:133]
    assert 0.25 < float(right_scores.mean()) < 0.45  # hovering at the threshold
    assert float(present["right"][lowered].mean()) > 0.5  # known leak, see the docstring


def test_masked_keypoints_of_a_real_clip_do_not_affect_the_output(
    raw_of: Callable[[str], RawKeypoints],
) -> None:
    """On hands_leave: overwrite the raw position of every masked, non-reference keypoint
    with garbage; the model inputs do not change."""
    raw = raw_of("hands_leave")
    before, mask = normalize_keypoints(
        raw.keypoints, raw.scores, raw.person_found, raw.frame_size, NORM
    )
    low = raw.scores <= np.float32(0.3)
    low &= raw.scores.astype(np.float64) <= 0.3
    low[:, [91, 112, 53]] = False  # the three reference points are the documented exception
    assert int(low[:, 91:133].sum()) > 500  # the lowered hand really is in there
    garbage = raw.keypoints.copy()
    garbage[low] = -9999.0
    after, mask_after = normalize_keypoints(
        garbage, raw.scores, raw.person_found, raw.frame_size, NORM
    )
    for part in before:
        np.testing.assert_array_equal(before[part], after[part])
        np.testing.assert_array_equal(mask[part], mask_after[part])


def test_blank_video_does_not_raise_and_presents_nothing(
    tmp_path: Path, core: PoseCore, video_writer: Callable[..., Path]
) -> None:
    black = [np.zeros((480, 640, 3), np.uint8)] * 12
    path = video_writer(tmp_path / "blank.mkv", black, [round(i * 1000 / 30) for i in range(12)])
    raw = extract_file(path, core)
    assert len(raw) == 12 and raw.frame_size == (640, 480)
    assert not raw.person_found.any()
    inputs, mask = normalize_keypoints(
        raw.keypoints, raw.scores, raw.person_found, raw.frame_size, NORM
    )
    print(
        "blank video: max score",
        float(raw.scores.max()),
        {p: float(m.mean()) for p, m in mask.items()},
    )
    assert all(not m.any() for m in mask.values())
    assert all(not v.any() for v in inputs.values())


def test_iter_keypoints_accepts_any_frame_source(
    core: PoseCore, clip_path: Callable[[str], Path]
) -> None:
    frames = [f for f, _ in zip(read_frames(clip_path("hands_visible")), range(3), strict=False)]
    out = list(iter_keypoints(frames, core))
    assert [o.t_ms for o in out] == [f.t_ms for f in frames]
    with pytest.raises(ValueError, match="BGR"):
        core.process(frames[0].image[..., 0], 0)
