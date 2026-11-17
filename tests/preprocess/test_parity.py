"""Parity with the base model's own pipeline, on the fixture clips.

Reference: the Uni-Sign code at the pinned commit, run unmodified in a separate process
(`reference_cli.py`): OpenCV decoding, its bundled and patched rtmlib 0.0.13, its
`load_part_kp` normalization, and its model.
Ours: PyAV decoding, rtmlib 0.0.16 with our own largest-box selection, and
`normalize_keypoints`.

Stated tolerances (the measured differences are about a thousand times smaller and come
from float32 storage of pixel coordinates; `docs/base_model_notes.md` has the numbers):

- raw keypoints: 1e-3 pixel; scores: 1e-6
- model inputs: 1e-6
- translated text: identical
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from signtoenglish.common.config import PathsConfig
from signtoenglish.preprocess.normalize import normalize_keypoints
from signtoenglish.preprocess.schema import PARTS, NormalizeConfig, RawKeypoints

KEYPOINT_TOL_PX = 1e-3
SCORE_TOL = 1e-6
INPUT_TOL = 1e-6
CLIPS = ["hands_visible", "hands_leave"]
NORM = NormalizeConfig()


def _ours(raw: RawKeypoints):  # type: ignore[no-untyped-def]
    return normalize_keypoints(raw.keypoints, raw.scores, raw.person_found, raw.frame_size, NORM)


@pytest.mark.parametrize("stem", CLIPS)
def test_raw_keypoints_match_the_reference_extraction(
    stem: str, raw_of: Callable[[str], RawKeypoints], reference_of: Callable[[str], dict[str, Path]]
) -> None:
    raw = raw_of(stem)
    ref = np.load(reference_of(stem)["extract"])
    assert str(ref["rtmlib_version"]) == "0.0.13"
    assert int(ref["n_people"].max()) == 1  # the reference keeps one person per frame
    assert (int(ref["width"]), int(ref["height"])) == raw.frame_size
    ref_px = ref["keypoints"][:, 0] * np.array(raw.frame_size, dtype=np.float64)
    ref_scores = ref["scores"][:, 0]
    assert ref_px.shape == raw.keypoints.shape  # same frames kept: the reference uses all of them
    kp_diff = float(np.abs(raw.keypoints - ref_px).max())
    score_diff = float(np.abs(raw.scores - ref_scores).max())
    print(
        f"{stem}: max keypoint difference {kp_diff:.2e} px, max score difference {score_diff:.2e}"
    )
    assert kp_diff <= KEYPOINT_TOL_PX
    assert score_diff <= SCORE_TOL


@pytest.mark.parametrize("stem", CLIPS)
def test_model_inputs_match_the_reference_pipeline(
    stem: str, raw_of: Callable[[str], RawKeypoints], reference_of: Callable[[str], dict[str, Path]]
) -> None:
    inputs, mask = _ours(raw_of(stem))
    ref = np.load(reference_of(stem)["inputs"])
    worst = 0.0
    for part in PARTS:
        assert inputs[part].shape == ref[part].shape
        worst = max(worst, float(np.abs(inputs[part].astype(np.float64) - ref[part]).max()))
        # Our mask is exactly "the reference did not zero this keypoint's score".
        np.testing.assert_array_equal(mask[part], ref[part][..., 2] != 0)
    print(f"{stem}: max model-input difference {worst:.2e}")
    assert worst <= INPUT_TOL


@pytest.mark.parametrize("stem", CLIPS)
def test_normalization_alone_equals_the_reference_function(
    stem: str,
    raw_of: Callable[[str], RawKeypoints],
    uni_sign_dir: Path,
    run_reference: Callable[..., None],
    tmp_path: Path,
) -> None:
    """Same raw keypoints into both normalizations, including every low-score and
    out-of-frame point of the clip: the float32 results are bit-identical."""
    raw = raw_of(stem)
    as_reference = tmp_path / "raw.npz"
    np.savez(
        as_reference,
        keypoints=(raw.keypoints.astype(np.float64) / np.array(raw.frame_size, dtype=np.float64))[
            :, None
        ],
        scores=raw.scores[:, None],
    )
    out = tmp_path / "inputs.npz"
    run_reference("normalize", uni_sign_dir, as_reference, out)
    ref = np.load(out)
    inputs, _ = _ours(raw)
    for part in PARTS:
        assert np.array_equal(inputs[part], ref[part].astype(np.float32)), part


def test_translated_text_is_identical_to_the_reference_pipeline(
    raw_of: Callable[[str], RawKeypoints],
    reference_of: Callable[[str], dict[str, Path]],
    uni_sign_dir: Path,
    paths: PathsConfig,
    need_file: Callable[[Path, str], Path],
    run_reference: Callable[..., None],
    tmp_path: Path,
) -> None:
    """Both pipelines' inputs through the pretrained model (OpenASL checkpoint, CPU, 32-bit
    floats, the reference's decoding settings): same token ids, same text, for both clips.
    The model is loaded in a separate process and freed when it exits."""
    checkpoint = need_file(
        paths.assets_root / "unisign" / "openasl_pose_only_slt.pth",
        "Uni-Sign checkpoint (asset A1)",
    )
    mt5 = need_file(
        paths.assets_root / "mt5-base" / "spiece.model", "mT5 tokenizer (asset A4)"
    ).parent
    files: list[Path] = []
    for stem in CLIPS:
        inputs, _ = _ours(raw_of(stem))
        ours = tmp_path / f"{stem}__ours.npz"
        np.savez(ours, **inputs)
        files += [ours, reference_of(stem)["inputs"]]
    out = tmp_path / "text.json"
    run_reference("translate", uni_sign_dir, mt5, checkpoint, "OpenASL", out, *files)
    results = json.loads(out.read_text(encoding="utf-8"))
    assert len(results) == 4
    for stem, ours, ref in zip(CLIPS, results[0::2], results[1::2], strict=True):
        print(f"{stem}: ours {ours['text']!r} | reference {ref['text']!r}")
        assert ours["frames"] == ref["frames"] == 150
        assert ours["text"].strip() != ""
        assert ours["ids"] == ref["ids"]
        assert ours["text"] == ref["text"]
