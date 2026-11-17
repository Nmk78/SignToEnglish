"""Loading for inference: the config file, and the refusals that must come before a
fine-tuned checkpoint is allowed to run."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from signtoenglish.common.hashing import config_hash, sha256_file
from signtoenglish.infer import loader
from signtoenglish.infer.loader import (
    CheckpointRefusedError,
    PreprocessingMismatchError,
    checkpoint_header,
    load_keypoint_config,
    load_model_config,
    load_translator,
    read_checkpoint_header,
)
from signtoenglish.models.translator import DecodeConfig, Translator
from signtoenglish.preprocess.cache import preprocessing_hash
from signtoenglish.preprocess.schema import KeypointConfig, NormalizeConfig


def write_checkpoint(
    path: Path, header: dict[str, Any], trainable: dict[str, torch.Tensor] | None = None
) -> Path:
    torch.save({**header, "trainable": trainable or {}, "step": 10}, path)
    return path


def never_load(*args: Any, **kwargs: Any) -> Translator:
    raise AssertionError(
        "the model was loaded although the checkpoint should have been refused first"
    )


def test_committed_configs_load_and_hold_the_reference_defaults(repo_root: Path) -> None:
    model_cfg = load_model_config(repo_root / "configs" / "model" / "unisign_pose.yaml")
    t = model_cfg.translator
    assert t.decode == DecodeConfig(num_beams=4, max_new_tokens=100)
    assert t.max_frames == 256 and t.hidden_dim == 256 and t.label_smoothing == 0.2
    assert model_cfg.normalize == NormalizeConfig() and model_cfg.device == "auto"
    assert set(t.checkpoints) == {"openasl", "how2sign"} and t.default_checkpoint == "openasl"
    assert {c.dataset for c in t.checkpoints.values()} == {"OpenASL", "How2Sign"}
    assert all(len(s) == 64 for s in [*t.reference_files.values(), *t.mt5_files.values()])
    assert len(config_hash(model_cfg)) == 16  # no path-typed field, so it can be hashed
    assert (
        load_keypoint_config(repo_root / "configs" / "preprocess" / "keypoints.yaml")
        == KeypointConfig()
    )


def test_checkpoint_from_other_preprocessing_is_refused_before_loading(
    tmp_path: Path, paths: Any, model_cfg: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PLAN.md Step 2.1: a checkpoint whose preprocessing hash differs from the runtime config
    is refused. Needs no model files: the refusal happens before anything is loaded."""
    monkeypatch.setattr(Translator, "load", never_load)
    trained_with = KeypointConfig()
    good = checkpoint_header(trained_with, model_cfg)
    assert good["preprocessing_hash"] == preprocessing_hash(trained_with)
    ckpt = write_checkpoint(tmp_path / "best.pt", good)

    for runtime in (
        trained_with.model_copy(update={"model_rate_fps": 15.0}),
        trained_with.model_copy(update={"pose_provider": "coreml"}),
    ):
        with pytest.raises(PreprocessingMismatchError, match="preprocessing hash") as info:
            load_translator(paths, model_cfg, runtime, fine_tuned=ckpt, device="cpu")
        assert good["preprocessing_hash"] in str(info.value) and preprocessing_hash(runtime) in str(
            info.value
        )

    stricter = model_cfg.model_copy(update={"normalize": NormalizeConfig(score_threshold=0.5)})
    with pytest.raises(CheckpointRefusedError, match="normalization settings"):
        load_translator(paths, stricter, trained_with, fine_tuned=ckpt, device="cpu")
    with pytest.raises(CheckpointRefusedError, match="base checkpoint"):
        load_translator(
            paths,
            model_cfg,
            trained_with,
            base_checkpoint="how2sign",
            fine_tuned=ckpt,
            device="cpu",
        )


def test_unusable_checkpoint_files_are_refused(tmp_path: Path, model_cfg: Any) -> None:
    header = checkpoint_header(KeypointConfig(), model_cfg)
    with pytest.raises(CheckpointRefusedError, match="not found"):
        read_checkpoint_header(tmp_path / "absent.pt")
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointRefusedError, match="cannot read"):
        read_checkpoint_header(junk)
    base_like = tmp_path / "base.pth"
    torch.save({"model": {"w": torch.zeros(2)}}, base_like)
    with pytest.raises(CheckpointRefusedError, match="not a s2e-checkpoint"):
        read_checkpoint_header(base_like)
    with pytest.raises(CheckpointRefusedError, match="version"):
        read_checkpoint_header(write_checkpoint(tmp_path / "v9.pt", {**header, "version": 9}))
    partial = {k: v for k, v in header.items() if k != "preprocessing_hash"}
    torch.save({**partial, "trainable": {}}, tmp_path / "partial.pt")
    with pytest.raises(CheckpointRefusedError, match="lacks"):
        read_checkpoint_header(tmp_path / "partial.pt")
    ok = read_checkpoint_header(write_checkpoint(tmp_path / "ok.pt", header))
    assert ok.preprocessing_hash == header["preprocessing_hash"] and ok.extra == {"step": 10}


def test_matching_fine_tuned_checkpoint_loads_on_top_of_the_base(
    tmp_path: Path,
    paths: Any,
    model_cfg: Any,
    keypoint_cfg: Any,
    model_for: Callable[..., Any],
    clip_inputs: Callable[[str], dict[str, np.ndarray]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint with the right header and plain model weights is applied, changes the
    output, and is identified by its own file hash. The already-loaded base model is reused
    here instead of loading a second copy."""
    base = model_for("openasl", "cpu")
    inputs = clip_inputs("hands_leave")
    before = base.translator.translate(inputs)[0]
    before_embeds, _ = base.translator.embed_for_test(inputs)
    module = base.translator.torch_module
    original = module.pose_proj.bias.detach().clone()
    monkeypatch.setattr(Translator, "load", lambda *a, **k: base.translator)
    header = checkpoint_header(keypoint_cfg, model_cfg)
    try:
        changed = write_checkpoint(tmp_path / "best.pt", header, {"pose_proj.bias": original + 3.0})
        loaded = load_translator(paths, model_cfg, keypoint_cfg, fine_tuned=changed, device="cpu")
        assert loaded.checkpoint_id == sha256_file(changed) != base.checkpoint_id
        assert (
            loaded.fine_tuned_from == changed
            and loaded.preprocessing_hash == header["preprocessing_hash"]
        )
        assert torch.equal(module.pose_proj.bias, original + 3.0)
        after_embeds, _ = loaded.translator.embed_for_test(inputs)
        assert not torch.allclose(after_embeds, before_embeds)  # it really changed
    finally:
        with torch.no_grad():
            module.pose_proj.bias.copy_(original)
    assert base.translator.translate(inputs)[0] == before

    adapter = write_checkpoint(tmp_path / "lora.pt", header, {"lora_A.weight": torch.zeros(2, 2)})
    with pytest.raises(CheckpointRefusedError, match=r"adapters\.py"):
        load_translator(paths, model_cfg, keypoint_cfg, fine_tuned=adapter, device="cpu")
    wrong_shape = write_checkpoint(
        tmp_path / "shape.pt", header, {"pose_proj.bias": torch.zeros(3)}
    )
    with pytest.raises(CheckpointRefusedError, match="wrong shape"):
        load_translator(paths, model_cfg, keypoint_cfg, fine_tuned=wrong_shape, device="cpu")
    assert torch.equal(module.pose_proj.bias, original)
    assert loader.CHECKPOINT_FORMAT == "s2e-checkpoint"
