"""Loading the translator for inference, with the checks that must pass before it runs.

- The model config (`configs/model/unisign_pose.yaml`) is read into `ModelConfig`: the
  translator's pinned files and decoding defaults, plus the normalization settings, which
  belong to the model because the pretrained weights dictate them.
- A base checkpoint (a Uni-Sign `.pth`) is pinned by SHA-256 in that config and verified
  by `Translator.load`.
- A fine-tuned checkpoint written by this project (`runs/<id>/best.pt`, Step 2.4) carries a
  header saying which preprocessing produced its training keypoints and which base
  checkpoint it sits on. `load_translator` REFUSES to run it when the runtime preprocessing
  hash, the normalization settings, or the base checkpoint differ. That refusal is the guard
  against a train/inference mismatch.

Nothing here imports training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signtoenglish.common.config import FrozenConfig, PathsConfig, find_repo_root, load_config
from signtoenglish.common.device import DevicePreference, select_device
from signtoenglish.common.hashing import config_hash, sha256_file
from signtoenglish.models.translator import Translator, TranslatorConfig, TranslatorError
from signtoenglish.preprocess.cache import preprocessing_hash
from signtoenglish.preprocess.schema import KeypointConfig, NormalizeConfig

DEFAULT_MODEL_CONFIG = Path("configs") / "model" / "unisign_pose.yaml"
DEFAULT_KEYPOINT_CONFIG = Path("configs") / "preprocess" / "keypoints.yaml"

#: Header format of a checkpoint written by this project. `train/checkpoint.py` (Step 2.4)
#: writes it; this module reads it.
CHECKPOINT_FORMAT = "s2e-checkpoint"
CHECKPOINT_VERSION = 1
HEADER_FIELDS = (
    "format",
    "version",
    "preprocessing_hash",
    "normalize_hash",
    "base_checkpoint_sha256",
)


class ModelConfig(FrozenConfig):
    """`configs/model/unisign_pose.yaml`."""

    translator: TranslatorConfig
    normalize: NormalizeConfig = NormalizeConfig()
    device: DevicePreference = "auto"


class CheckpointRefusedError(RuntimeError):
    """A fine-tuned checkpoint does not match the runtime configuration and was not loaded."""


class PreprocessingMismatchError(CheckpointRefusedError):
    """The checkpoint was trained on keypoints from a different preprocessing configuration."""


@dataclass(frozen=True, slots=True)
class CheckpointHeader:
    preprocessing_hash: str
    normalize_hash: str
    base_checkpoint_sha256: str
    trainable: dict[str, Any]
    extra: dict[str, Any]


def load_model_config(path: str | Path | None = None) -> ModelConfig:
    return load_config(Path(path) if path else find_repo_root() / DEFAULT_MODEL_CONFIG, ModelConfig)


def load_keypoint_config(path: str | Path | None = None) -> KeypointConfig:
    return load_config(
        Path(path) if path else find_repo_root() / DEFAULT_KEYPOINT_CONFIG, KeypointConfig
    )


def checkpoint_header(
    keypoint_cfg: KeypointConfig, model_cfg: ModelConfig, base_checkpoint: str | None = None
) -> dict[str, Any]:
    """The header a checkpoint trained under these configs must carry. Training code puts
    this dict, plus `trainable` and whatever else it needs, into the file it saves."""
    name = base_checkpoint or model_cfg.translator.default_checkpoint
    if name not in model_cfg.translator.checkpoints:
        raise TranslatorError(
            f"unknown checkpoint {name!r}; configured: {sorted(model_cfg.translator.checkpoints)}"
        )
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "preprocessing_hash": preprocessing_hash(keypoint_cfg),
        "normalize_hash": config_hash(model_cfg.normalize),
        "base_checkpoint_sha256": model_cfg.translator.checkpoints[name].sha256,
    }


def read_checkpoint_header(path: str | Path) -> CheckpointHeader:
    """Read and validate the header of a fine-tuned checkpoint. Tensors only; no code runs."""
    import torch

    path = Path(path)
    if not path.is_file():
        raise CheckpointRefusedError(f"checkpoint not found: {path}")
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # unreadable or not a plain tensor file
        raise CheckpointRefusedError(f"cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(blob, dict) or blob.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointRefusedError(
            f"{path} is not a {CHECKPOINT_FORMAT} file (a Uni-Sign base checkpoint is selected "
            "by name in the model config, not passed here)"
        )
    if blob.get("version") != CHECKPOINT_VERSION:
        raise CheckpointRefusedError(
            f"{path}: unsupported checkpoint version {blob.get('version')!r}"
        )
    missing = [k for k in (*HEADER_FIELDS, "trainable") if k not in blob]
    if missing:
        raise CheckpointRefusedError(f"{path}: checkpoint header lacks {missing}")
    trainable = blob["trainable"]
    if not isinstance(trainable, dict):
        raise CheckpointRefusedError(f"{path}: 'trainable' must be a dict of tensors")
    return CheckpointHeader(
        preprocessing_hash=str(blob["preprocessing_hash"]),
        normalize_hash=str(blob["normalize_hash"]),
        base_checkpoint_sha256=str(blob["base_checkpoint_sha256"]),
        trainable=trainable,
        extra={k: v for k, v in blob.items() if k not in (*HEADER_FIELDS, "trainable")},
    )


@dataclass(frozen=True, slots=True)
class LoadedModel:
    translator: Translator
    model_cfg: ModelConfig
    keypoint_cfg: KeypointConfig
    #: What identifies the weights in use: the fine-tuned file's SHA-256 if one was loaded,
    #: otherwise the base checkpoint's.
    checkpoint_id: str
    preprocessing_hash: str
    fine_tuned_from: Path | None


def load_translator(
    paths: PathsConfig,
    model_cfg: ModelConfig,
    keypoint_cfg: KeypointConfig,
    *,
    base_checkpoint: str | None = None,
    fine_tuned: str | Path | None = None,
    device: DevicePreference | None = None,
) -> LoadedModel:
    """Load the base model and, if given, a fine-tuned checkpoint on top of it.

    Checks happen before any weights are loaded, so a refusal is cheap and a refused
    checkpoint never produces output.
    """
    import torch

    name = base_checkpoint or model_cfg.translator.default_checkpoint
    runtime = checkpoint_header(keypoint_cfg, model_cfg, name)
    header: CheckpointHeader | None = None
    if fine_tuned is not None:
        header = read_checkpoint_header(fine_tuned)
        if header.preprocessing_hash != runtime["preprocessing_hash"]:
            raise PreprocessingMismatchError(
                f"{fine_tuned} was trained on keypoints with preprocessing hash "
                f"{header.preprocessing_hash}; the runtime configuration has "
                f"{runtime['preprocessing_hash']}. Refusing to run: its inputs would not match "
                "what it was trained on."
            )
        if header.normalize_hash != runtime["normalize_hash"]:
            raise CheckpointRefusedError(
                f"{fine_tuned} was trained with normalization settings {header.normalize_hash}; "
                f"the runtime configuration has {runtime['normalize_hash']}"
            )
        if header.base_checkpoint_sha256 != runtime["base_checkpoint_sha256"]:
            raise CheckpointRefusedError(
                f"{fine_tuned} sits on base checkpoint {header.base_checkpoint_sha256}; "
                f"the selected base checkpoint {name!r} is {runtime['base_checkpoint_sha256']}"
            )

    chosen = select_device(device or model_cfg.device)
    translator = Translator.load(
        model_cfg.translator, paths.assets_root, paths.third_party_root, chosen, checkpoint=name
    )
    checkpoint_id = runtime["base_checkpoint_sha256"]
    if header is not None:
        module = translator.torch_module
        state = module.state_dict()
        unknown = sorted(set(header.trainable) - set(state))
        if unknown:
            raise CheckpointRefusedError(
                f"{fine_tuned}: {len(unknown)} weights are not parameters of the base model "
                f"(first: {unknown[:3]}). Adapter weights are loaded by models/adapters.py "
                "(Step 2.4), which does not exist yet."
            )
        for key, value in header.trainable.items():
            if not isinstance(value, torch.Tensor) or value.shape != state[key].shape:
                raise CheckpointRefusedError(
                    f"{fine_tuned}: weight {key!r} has the wrong shape or type"
                )
        module.load_state_dict(
            {k: v.to(state[k].dtype) for k, v in header.trainable.items()}, strict=False
        )
        checkpoint_id = sha256_file(fine_tuned)  # type: ignore[arg-type]
    return LoadedModel(
        translator=translator,
        model_cfg=model_cfg,
        keypoint_cfg=keypoint_cfg,
        checkpoint_id=checkpoint_id,
        preprocessing_hash=runtime["preprocessing_hash"],
        fine_tuned_from=Path(fine_tuned) if fine_tuned is not None else None,
    )
