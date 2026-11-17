"""The translator wrapper: the real Uni-Sign model, loaded from the pinned local files, on
the fixture clips. See tests/preprocess/conftest.py for what happens where files are absent.

Tests are ordered so that each (checkpoint, device) pair is loaded once: OpenASL on CPU
first, then OpenASL on MPS, then How2Sign on CPU.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from signtoenglish.models.translator import (
    PART_SIZES,
    PARTS,
    DecodeConfig,
    Translator,
    TranslatorError,
)

CLIPS = ("hands_visible", "hands_leave")
Inputs = Callable[[str], dict[str, np.ndarray]]


# --- pins: refused before anything is loaded (cheap) -------------------------------------


def _load(cfg: Any, paths: Any, **kw: Any) -> Translator:
    return Translator.load(cfg, paths.assets_root, paths.third_party_root, "cpu", **kw)


def test_wrong_pins_are_refused(model_cfg: Any, paths: Any, model_files: None) -> None:
    t = model_cfg.translator
    with pytest.raises(TranslatorError, match="pinned commit"):
        _load(t.model_copy(update={"reference_commit": "0" * 40}), paths)
    bad_code = {**t.reference_files, "models.py": "0" * 64}
    with pytest.raises(TranslatorError, match="reference code"):
        _load(t.model_copy(update={"reference_files": bad_code}), paths)
    bad_tok = {**t.mt5_files, "spiece.model": "0" * 64}
    with pytest.raises(TranslatorError, match="tokenizer"):
        _load(t.model_copy(update={"mt5_files": bad_tok}), paths)
    wrong = t.checkpoints["openasl"].model_copy(update={"sha256": "0" * 64})
    with pytest.raises(TranslatorError, match="checkpoint"):
        _load(t.model_copy(update={"checkpoints": {"openasl": wrong}}), paths)
    chinese = t.checkpoints["openasl"].model_copy(update={"dataset": "CSL_Daily"})
    with pytest.raises(TranslatorError, match="Chinese"):
        _load(t.model_copy(update={"checkpoints": {"openasl": chinese}}), paths)
    with pytest.raises(TranslatorError, match="unknown checkpoint"):
        _load(t, paths, checkpoint="nope")


# --- OpenASL on CPU ----------------------------------------------------------------------


def test_loaded_model_is_float32_whole_and_laid_out_as_trained(
    model_for: Callable[..., Any],
) -> None:
    translator = model_for("openasl", "cpu").translator
    report = translator.report
    module = translator.torch_module
    mt5 = module.mt5_model
    assert report.checkpoint_tensors == 627
    assert report.mt5_parameters == 582_401_280  # mT5-Base, as in the Uni-Sign paper
    assert report.parameters == 587_747_368
    assert {p.dtype for p in module.parameters()} == {torch.float32}  # no bfloat16
    assert {p.device.type for p in module.parameters()} == {"cpu"}  # and no CUDA anywhere
    assert not module.training
    # The layout the checkpoint was trained with: one embedding table read by encoder and
    # decoder, and a separate output table.
    assert mt5.encoder.embed_tokens.weight is mt5.shared.weight
    assert mt5.decoder.embed_tokens.weight is mt5.shared.weight
    assert mt5.lm_head.weight is not mt5.shared.weight
    assert translator.prompt_ids() == [89349, 11329, 17896, 1552, 288, 5413, 267, 1]
    assert module.lang == "English"


def test_every_tensor_equals_the_checkpoint(
    model_for: Callable[..., Any], model_cfg: Any, paths: Any
) -> None:
    """The permanent guard against a silently wrong model. `transformers` 5.23.0 joins mT5's
    output layer to its embedding; the checkpoint holds different values for the two, so a
    joined model cannot equal it. Loaded our way, every tensor does."""
    translator = model_for("openasl", "cpu").translator
    ref = model_cfg.translator.checkpoints["openasl"]
    state = torch.load(
        paths.assets_root / ref.relative_path, map_location="cpu", weights_only=True
    )["model"]
    assert {str(v.dtype) for v in state.values()} == {"torch.bfloat16", "torch.int64"}
    shared, head = state["mt5_model.shared.weight"], state["mt5_model.lm_head.weight"]
    assert not torch.equal(shared, head)  # which is why they must be separate tensors
    model_state = translator.torch_module.state_dict()
    assert set(model_state) == set(state)
    differ = [k for k, v in model_state.items() if not torch.equal(v, state[k].to(v.dtype))]
    assert differ == []


@pytest.mark.parametrize("stem", CLIPS)
def test_translate_equals_the_reference_two_step_path(
    stem: str, model_for: Callable[..., Any], clip_inputs: Inputs
) -> None:
    """`translate` skips the reference's extra mT5 pass; the output must not change."""
    translator = model_for("openasl", "cpu").translator
    inputs = clip_inputs(stem)
    text, ids = translator.translate_with_ids(inputs)[0]
    ref_text, ref_ids = translator.translate_reference_path(inputs)
    print(f"{stem}: {text!r}")
    assert text.strip() and ids == ref_ids and text == ref_text
    assert translator.translate(inputs) == [text]  # and it is repeatable
    assert ids[0] == 0 and ids[-1] == 1  # decoder start ... end of sentence


def test_batches_are_decoded_singly_so_padding_cannot_change_outputs(
    model_for: Callable[..., Any], clip_inputs: Inputs
) -> None:
    translator = model_for("openasl", "cpu").translator
    long = clip_inputs("hands_visible")
    short = {k: v[:90] for k, v in clip_inputs("hands_leave").items()}
    singles = [translator.translate_with_ids(c)[0] for c in (long, short, long)]
    together = translator.translate_with_ids([long, short, long])
    assert together == singles
    assert translator.translate([short, long]) == [singles[1][0], singles[0][0]]


def test_what_the_base_models_padding_does_to_a_shorter_clip(
    model_for: Callable[..., Any], clip_inputs: Inputs
) -> None:
    """Measured, to explain the rule above. The reference pads a batch by repeating each
    clip's last frame. Its three temporal convolutions (kernel 5) reach six frames back, so
    the pose features of the last six real frames of a padded clip change; earlier frames
    and the prompt do not. Equal-length clips are not padded and are unaffected."""
    translator = model_for("openasl", "cpu").translator
    long = clip_inputs("hands_visible")
    short = {k: v[:90] for k, v in clip_inputs("hands_leave").items()}
    alone, _ = translator.embed_for_test(short)
    batch, mask = translator.embed_for_test([long, short])
    n_prompt = alone.shape[1] - 90
    assert n_prompt == 8 and int(mask[1].sum()) == alone.shape[1]
    diff = (alone[0] - batch[1, : alone.shape[1]]).abs().amax(dim=1).numpy()
    changed = np.nonzero(diff[n_prompt:] > 1e-4)[0].tolist()
    print(f"padding changes frames {changed} of 90; largest change {diff.max():.2f} "
          f"(mean feature size {float(alone.abs().mean()):.2f})")  # fmt: skip
    assert diff[:n_prompt].max() == 0
    assert changed == [84, 85, 86, 87, 88, 89]
    same, _ = translator.embed_for_test([long, long])
    one, _ = translator.embed_for_test(long)
    assert torch.allclose(same[0], one[0], atol=1e-5) and torch.allclose(same[1], one[0], atol=1e-5)


def test_bad_inputs_are_refused(model_for: Callable[..., Any], clip_inputs: Inputs) -> None:
    translator = model_for("openasl", "cpu").translator
    clip = clip_inputs("hands_leave")
    too_long = {k: np.concatenate([v, v])[:257] for k, v in clip.items()}
    with pytest.raises(TranslatorError, match="maximum is 256"):
        translator.translate(too_long)
    with pytest.raises(TranslatorError, match="exactly the groups"):
        translator.translate({k: v for k, v in clip.items() if k != "left"})
    with pytest.raises(TranslatorError, match="float32"):
        translator.translate({**clip, "body": clip["body"].astype(np.float64)})
    with pytest.raises(TranslatorError, match="no frames"):
        translator.translate({k: v[:0] for k, v in clip.items()})
    assert translator.translate([]) == []
    assert PARTS == ("body", "left", "right", "face_all")
    assert {k: v.shape[1] for k, v in clip.items()} == PART_SIZES


def test_decoding_settings_are_applied(model_for: Callable[..., Any], clip_inputs: Inputs) -> None:
    translator = model_for("openasl", "cpu").translator
    assert translator.cfg.decode == DecodeConfig(num_beams=4, max_new_tokens=100)
    _, ids = translator.translate_with_ids(
        clip_inputs("hands_visible"), DecodeConfig(num_beams=1, max_new_tokens=5)
    )[0]
    assert len(ids) <= 6  # the start token plus at most five new ones


def test_forward_returns_the_reference_training_loss(
    model_for: Callable[..., Any], clip_inputs: Inputs
) -> None:
    translator = model_for("openasl", "cpu").translator
    clips = [clip_inputs("hands_visible"), clip_inputs("hands_leave")]
    with torch.no_grad():
        loss = translator.forward(clips, ["Wash your hands.", "It's very beautiful."])
        own_output = translator.forward(clips[1], [translator.translate(clips[1])[0]])
        unrelated = translator.forward(clips[1], ["Quarterly revenue exceeded forecasts."])
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert float(own_output) < float(unrelated)  # its own translation is the likelier target
    with pytest.raises(TranslatorError, match="label sentences"):
        translator.forward(clips, ["one sentence for two clips"])
    assert translator.forward(clips[1], ["Hello."]).requires_grad  # usable by training code
    print(
        f"loss: batch {float(loss):.3f}, own {float(own_output):.3f}, other {float(unrelated):.3f}"
    )


@pytest.fixture(scope="module")
def cpu_results(model_for: Callable[..., Any], clip_inputs: Inputs) -> dict[str, Any]:
    translator = model_for("openasl", "cpu").translator
    out: dict[str, Any] = {}
    for stem in CLIPS:
        out[stem] = translator.translate_with_ids(clip_inputs(stem))[0]
    with torch.no_grad():
        out["loss"] = float(
            translator.forward(clip_inputs("hands_leave"), ["It's very beautiful."])
        )
    return out


def test_wrapper_equals_the_reference_pipeline_openasl(
    cpu_results: dict[str, Any],
    reference_of: Callable[[str], dict[str, Path]],
    uni_sign_dir: Path,
    paths: Any,
    model_cfg: Any,
    run_reference: Callable[..., None],
    tmp_path: Path,
) -> None:
    _compare_with_reference(
        "openasl",
        cpu_results,
        reference_of,
        uni_sign_dir,
        paths,
        model_cfg,
        run_reference,
        tmp_path,
    )


# --- OpenASL on MPS ----------------------------------------------------------------------


def test_forward_and_generate_on_mps_agree_with_cpu(
    cpu_results: dict[str, Any], model_for: Callable[..., Any], clip_inputs: Inputs
) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("LOCAL FILES ABSENT: this machine has no MPS device")
    translator = model_for("openasl", "mps").translator
    assert {p.device.type for p in translator.torch_module.parameters()} == {"mps"}
    assert {p.dtype for p in translator.torch_module.parameters()} == {torch.float32}
    for stem in CLIPS:
        text, ids = translator.translate_with_ids(clip_inputs(stem))[0]
        print(f"{stem} on MPS: {text!r}")
        assert (text, ids) == cpu_results[stem]
    with torch.no_grad():
        loss = float(translator.forward(clip_inputs("hands_leave"), ["It's very beautiful."]))
    print(f"loss on MPS {loss:.5f}, on CPU {cpu_results['loss']:.5f}")
    assert abs(loss - cpu_results["loss"]) < 1e-3


# --- How2Sign on CPU ---------------------------------------------------------------------


def test_wrapper_equals_the_reference_pipeline_how2sign(
    model_for: Callable[..., Any],
    clip_inputs: Inputs,
    reference_of: Callable[[str], dict[str, Path]],
    uni_sign_dir: Path,
    paths: Any,
    model_cfg: Any,
    run_reference: Callable[..., None],
    tmp_path: Path,
) -> None:
    translator = model_for("how2sign", "cpu").translator
    assert translator.report.checkpoint_name == "how2sign"
    ours = {stem: translator.translate_with_ids(clip_inputs(stem))[0] for stem in CLIPS}
    for stem in CLIPS:
        assert ours[stem][1] == translator.translate_reference_path(clip_inputs(stem))[1]
    _compare_with_reference(
        "how2sign", ours, reference_of, uni_sign_dir, paths, model_cfg, run_reference, tmp_path
    )


def _compare_with_reference(
    name: str,
    ours: dict[str, Any],
    reference_of: Callable[[str], dict[str, Path]],
    uni_sign_dir: Path,
    paths: Any,
    model_cfg: Any,
    run_reference: Callable[..., None],
    tmp_path: Path,
) -> None:
    """The whole reference pipeline (its extraction, its normalization, its model, in its own
    process) against extraction + normalization + wrapper here: same token ids, same text."""
    ref = model_cfg.translator.checkpoints[name]
    out = tmp_path / "reference.json"
    run_reference(
        "translate", uni_sign_dir, paths.assets_root / model_cfg.translator.mt5_dir,
        paths.assets_root / ref.relative_path, ref.dataset, out,
        *[reference_of(stem)["inputs"] for stem in CLIPS],
    )  # fmt: skip
    results = json.loads(out.read_text(encoding="utf-8"))
    for stem, reference in zip(CLIPS, results, strict=True):
        text, ids = ours[stem]
        print(f"{name} / {stem}: {text!r}")
        assert ids == reference["ids"], (
            f"{name}/{stem}: wrapper {text!r} vs reference {reference['text']!r}"
        )
        assert text == reference["text"]
