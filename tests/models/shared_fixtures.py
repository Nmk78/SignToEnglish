"""Fixtures shared by the model and inference tests (tests/models, tests/infer).

Loaded once per test process by both folders' `conftest.py` (by file path, because pytest's
importlib mode does not let test folders import each other). It re-exports the preprocessing
fixtures (pose core, fixture clips, the reference runner, and the rule for missing local
files; see tests/preprocess/conftest.py) and adds the translator.

Memory rule for this 16 GB machine: `model_for` keeps ONE translator loaded at a time. Asking
for another checkpoint or device frees the previous one first. Tests are written so that
each (checkpoint, device) pair is needed in one stretch.
"""

from __future__ import annotations

import gc
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_PREPROCESS_CONFTEST = Path(__file__).resolve().parents[1] / "preprocess" / "conftest.py"
_NAME = "s2e_shared_preprocess_fixtures"
if _NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_NAME, _PREPROCESS_CONFTEST)
    assert _spec is not None and _spec.loader is not None
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _module
    _spec.loader.exec_module(_module)
_pre = sys.modules[_NAME]

# Re-exported fixtures from the preprocessing tests.
paths = _pre.paths
keypoint_cfg = _pre.keypoint_cfg
core = _pre.core
clip_path = _pre.clip_path
raw_of = _pre.raw_of
uni_sign_dir = _pre.uni_sign_dir
run_reference = _pre.run_reference
reference_of = _pre.reference_of
need_file = _pre.need_file
video_writer = _pre.video_writer
need = _pre.need

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPS = ("hands_visible", "hands_leave")


@pytest.fixture(scope="session")
def model_cfg() -> Any:
    from signtoenglish.infer.loader import load_model_config

    return load_model_config(REPO_ROOT / "configs" / "model" / "unisign_pose.yaml")


@pytest.fixture(scope="session")
def model_files(model_cfg: Any, paths: Any) -> None:
    """Fail or skip (by the shared rule) unless everything the translator needs is local."""
    t = model_cfg.translator
    need(
        paths.third_party_root / t.reference_dir / "models.py",
        "Uni-Sign reference checkout (asset A3)",
    )
    need(
        paths.assets_root / t.mt5_dir / "spiece.model", "mT5 tokenizer and configuration (asset A4)"
    )
    for name, ref in t.checkpoints.items():
        need(paths.assets_root / ref.relative_path, f"Uni-Sign checkpoint {name!r} (assets A1, A2)")


@pytest.fixture(scope="session")
def model_for(
    model_cfg: Any, keypoint_cfg: Any, paths: Any, model_files: None
) -> Callable[..., Any]:
    """`model_for(checkpoint, device)` -> `LoadedModel`, one loaded at a time."""
    from signtoenglish.infer.loader import load_translator

    held: dict[tuple[str, str], Any] = {}
    loads: list[tuple[str, str]] = []

    def get(checkpoint: str = "openasl", device: str = "cpu") -> Any:
        key = (checkpoint, device)
        if key not in held:
            held.clear()
            gc.collect()
            if device != "mps":
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            held[key] = load_translator(
                paths,
                model_cfg,
                keypoint_cfg,
                base_checkpoint=checkpoint,
                device=device,  # type: ignore[arg-type]
            )
            loads.append(key)
        return held[key]

    get.loads = loads  # type: ignore[attr-defined]
    yield get
    held.clear()
    gc.collect()


@pytest.fixture(scope="session")
def clip_inputs(
    raw_of: Callable[[str], Any], model_cfg: Any
) -> Callable[[str], dict[str, np.ndarray]]:
    """Model inputs of a fixture clip, from real extraction and the shared normalization."""
    from signtoenglish.preprocess.normalize import normalize_keypoints

    cache: dict[str, dict[str, np.ndarray]] = {}

    def get(stem: str) -> dict[str, np.ndarray]:
        if stem not in cache:
            raw = raw_of(stem)
            inputs, _ = normalize_keypoints(
                raw.keypoints, raw.scores, raw.person_found, raw.frame_size, model_cfg.normalize
            )
            cache[stem] = {str(k): v for k, v in inputs.items()}
        return cache[stem]

    return get


FIXTURE_NAMES = (
    "paths", "keypoint_cfg", "core", "clip_path", "raw_of", "uni_sign_dir", "run_reference",
    "reference_of", "need_file", "video_writer", "model_cfg", "model_files", "model_for",
    "clip_inputs",
)  # fmt: skip
