"""The pretrained translator behind one interface.

This is the only module that imports the Uni-Sign code. That code has no license; it lives
in the git-ignored `third_party/` folder at a pinned commit, is never copied into `src/`,
and is used here unmodified: this module calls its classes and methods, it does not
re-implement them.

What this wrapper adds around the reference model (`docs/base_model_notes.md`,
`docs/feasibility.md`):

- It runs in 32-bit floats on MPS or CPU. The reference script hardcodes CUDA and bfloat16.
- mT5 is built from the local `config.json`; no pretrained mT5 weight file is needed,
  because the Uni-Sign checkpoint supplies every tensor. That is checked on every load:
  after loading, every tensor in the model must equal the checkpoint's tensor of that name.
- mT5's output layer is separated from the input embedding before loading. `transformers`
  5.23.0 joins them whatever `config.json` says; these checkpoints were trained with them
  apart and hold different values. Loaded joined, the embedding is silently overwritten and
  the model returns nothing. The equality check above is what catches it.
- The prompt asks for English (`dataset` is an ASL dataset name, never the reference's
  `CSL_Daily` default).
- `translate` does not run the reference's extra mT5 pass. The reference calls
  `model(src, tgt)`, which runs the pose encoder AND a full mT5 forward with a loss, and
  then `generate`. Here the reference's own `forward` is entered as written and stopped by
  a hook at the moment it hands the embeddings to mT5, so the embeddings are the
  reference's, bit for bit, without the wasted pass. A test compares with the full path.

Interface (ARCHITECTURE.md, section 4.6):

    Translator.translate(model_inputs, decode_cfg) -> list[str]
    Translator.forward(model_inputs, labels) -> loss      # the reference's own training loss

`model_inputs` is one clip or a list of clips; a clip is the dict of four float32 arrays
`[T, V, 3]` that `preprocess.normalize.normalize_keypoints` returns. The mask it also
returns is not a model input (the base model has none); see `normalize.py`.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import time
import types
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from signtoenglish.common.config import FrozenConfig
from signtoenglish.common.device import DeviceName
from signtoenglish.common.hashing import HashMismatchError, verify_sha256

# Model hubs stay offline: every file is local and pinned (CLAUDE.md, stop point 2).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

#: Input groups in the base model's order (`Uni_Sign.modes`), with keypoints per group.
PARTS: tuple[str, ...] = ("body", "left", "right", "face_all")
PART_SIZES: dict[str, int] = {"body": 9, "left": 21, "right": 21, "face_all": 18}

ClipInputs = Mapping[str, npt.NDArray[np.float32]]

_REFERENCE_MODULES = ("config", "stgcn_layers", "deformable_attention_2d", "models")


class TranslatorError(RuntimeError):
    """The model could not be built or loaded as pinned, or was given unusable input."""


class DecodeConfig(FrozenConfig):
    """Decoding settings. Defaults are the reference's (`fine_tuning.py:263-266`); nothing
    else is set there, so everything else is the library default for mT5."""

    num_beams: int = 4
    max_new_tokens: int = 100


class CheckpointRef(FrozenConfig):
    """One Uni-Sign checkpoint under the assets root, pinned by content hash."""

    relative_path: str
    sha256: str
    #: The dataset the checkpoint was fine-tuned on. It sets the prompt language: any name
    #: without "CSL" means English (`models.py:104-107`).
    dataset: str


class TranslatorConfig(FrozenConfig):
    """What defines the translator; the values live in `configs/model/unisign_pose.yaml`.
    Locations are relative to the assets and third-party roots; the SHA-256 values are what
    pin the content."""

    #: Uni-Sign checkout: folder under the third-party root, its commit, and file hashes.
    reference_dir: str
    reference_commit: str
    reference_files: dict[str, str]
    #: mT5-Base configuration and tokenizer files (asset A4): folder under the assets root.
    mt5_dir: str
    mt5_files: dict[str, str]
    checkpoints: dict[str, CheckpointRef]
    default_checkpoint: str
    #: The reference's `--hidden_dim` and `--label_smoothing` defaults (`utils.py:455,532`).
    hidden_dim: int = 256
    label_smoothing: float = 0.2
    #: The base model's training-time maximum (`utils.py:523`). `translate` refuses more; the
    #: caller thins longer inputs first (see `infer/offline.py`).
    max_frames: int = 256
    #: mT5-Base with a separate output table: 582.4 M, as in the Uni-Sign paper (Table 12).
    mt5_parameters: int = 582_401_280
    #: The prompt the checkpoint expects, as token ids. Guards against a tokenizer change.
    expected_prompt_ids: tuple[int, ...]
    decode: DecodeConfig = DecodeConfig()


@dataclass(frozen=True, slots=True)
class LoadReport:
    """What happened while loading; kept for `/health`, logs and tests."""

    checkpoint_name: str
    checkpoint_sha256: str
    device: str
    parameters: int
    mt5_parameters: int
    checkpoint_tensors: int
    output_layer_was_joined: bool
    seconds: float


class _StopAtMt5(Exception):
    """Raised by the capture hook to leave the reference forward before the mT5 pass."""


def _verify_files(root: Path, files: Mapping[str, str], what: str) -> None:
    for name, sha in files.items():
        path = root / name
        if not path.is_file():
            raise TranslatorError(
                f"{what}: {path} not found (a pinned local file; never downloaded)"
            )
        try:
            verify_sha256(path, sha)
        except HashMismatchError as exc:
            raise TranslatorError(f"{what}: {exc}") from exc


def _checkout_commit(reference_root: Path) -> str | None:
    head = reference_root / ".git" / "HEAD"
    try:
        return head.read_text(encoding="utf-8").strip()
    except OSError:
        return None


@contextlib.contextmanager
def _reference_on_path(reference_root: Path) -> Iterator[None]:
    """The one `sys.path` edit in the project: the Uni-Sign checkout, while it is imported."""
    entry = str(reference_root)
    sys.path.insert(0, entry)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(entry)


def _import_reference(reference_root: Path) -> Any:
    """Import the reference `models` module from the pinned checkout and nowhere else."""
    resolved = reference_root.resolve()
    for name in _REFERENCE_MODULES:
        loaded = sys.modules.get(name)
        origin = getattr(loaded, "__file__", None)
        if loaded is not None and (
            origin is None or not Path(origin).resolve().is_relative_to(resolved)
        ):
            raise TranslatorError(
                f"a module named {name!r} is already imported from {origin}; it would shadow "
                "the Uni-Sign reference module of the same name"
            )
    with _reference_on_path(reference_root):
        module = importlib.import_module("models")
    if not Path(str(module.__file__)).resolve().is_relative_to(resolved):
        raise TranslatorError(f"imported `models` from {module.__file__}, not from {resolved}")
    return module


class _Mt5FromConfig:
    """Stands in for `MT5ForConditionalGeneration` in the reference module's namespace while
    `Uni_Sign.__init__` runs: builds mT5 from the local config, without pretrained weights and
    without spending time on random initial values that the checkpoint then overwrites."""

    @staticmethod
    def from_pretrained(path: str, *args: Any, **kwargs: Any) -> Any:
        from transformers import MT5Config, MT5ForConditionalGeneration
        from transformers.initialization import no_init_weights

        config = MT5Config.from_pretrained(path, local_files_only=True)
        with no_init_weights():
            return MT5ForConditionalGeneration(config)


class Translator:
    """The Uni-Sign pose-only model, loaded and ready on one device."""

    def __init__(
        self, model: Any, cfg: TranslatorConfig, device: DeviceName, report: LoadReport
    ) -> None:
        self._model = model
        self.cfg = cfg
        self.device: DeviceName = device
        self.report = report

    # --- loading -----------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        cfg: TranslatorConfig,
        assets_root: Path,
        third_party_root: Path,
        device: DeviceName,
        checkpoint: str | None = None,
    ) -> Translator:
        """Build the reference model, load a pinned checkpoint, and prove it loaded whole."""
        import torch

        started = time.perf_counter()
        name = checkpoint or cfg.default_checkpoint
        if name not in cfg.checkpoints:
            raise TranslatorError(
                f"unknown checkpoint {name!r}; configured: {sorted(cfg.checkpoints)}"
            )
        ref = cfg.checkpoints[name]
        if "CSL" in ref.dataset:
            raise TranslatorError(f"checkpoint dataset {ref.dataset!r} would prompt for Chinese")

        reference_root = third_party_root / cfg.reference_dir
        commit = _checkout_commit(reference_root)
        if commit != cfg.reference_commit:
            raise TranslatorError(
                f"Uni-Sign checkout at {reference_root} is at commit {commit!r}, "
                f"pinned commit is {cfg.reference_commit}"
            )
        _verify_files(reference_root, cfg.reference_files, "Uni-Sign reference code")
        mt5_dir = assets_root / cfg.mt5_dir
        _verify_files(mt5_dir, cfg.mt5_files, "mT5 configuration and tokenizer")
        checkpoint_path = assets_root / ref.relative_path
        _verify_files(
            checkpoint_path.parent, {checkpoint_path.name: ref.sha256}, "Uni-Sign checkpoint"
        )

        reference = _import_reference(reference_root)
        args = types.SimpleNamespace(
            hidden_dim=cfg.hidden_dim,
            dataset=ref.dataset,
            rgb_support=False,  # pose-only; the RGB branch would fetch weights and needs CUDA
            label_smoothing=cfg.label_smoothing,
        )
        saved = (reference.MT5ForConditionalGeneration, reference.mt5_path)
        reference.MT5ForConditionalGeneration = _Mt5FromConfig
        reference.mt5_path = str(mt5_dir)
        try:
            model = reference.Uni_Sign(args=args)
        finally:
            reference.MT5ForConditionalGeneration, reference.mt5_path = saved

        # Put mT5's weight sharing into the layout the checkpoint was trained with
        # (transformers 4.40): encoder and decoder read the one shared embedding table, and
        # the output layer has a table of its own. Neither holds by itself here: building
        # without initialization skips the library's tying, and when the library does tie,
        # it joins the output layer to the embedding as well.
        mt5 = model.mt5_model
        joined = mt5.lm_head.weight is mt5.shared.weight
        mt5.config.tie_word_embeddings = False
        mt5.encoder.embed_tokens = mt5.shared
        mt5.decoder.embed_tokens = mt5.shared
        mt5.lm_head.weight = torch.nn.Parameter(torch.empty_like(mt5.shared.weight))
        mt5_parameters = sum(p.numel() for p in mt5.parameters())
        if mt5_parameters != cfg.mt5_parameters:
            raise TranslatorError(
                f"mT5 has {mt5_parameters} parameters after arranging weight sharing, "
                f"expected {cfg.mt5_parameters}"
            )

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise TranslatorError(f"checkpoint {name!r} does not fit the model: {exc}") from exc
        differ = [
            k for k, v in model.state_dict().items() if not torch.equal(v, state[k].to(v.dtype))
        ]
        if differ:
            raise TranslatorError(
                f"{len(differ)} tensors differ from checkpoint {name!r} after loading "
                f"(first: {differ[:3]}); the model would be silently wrong"
            )
        n_tensors = len(state)
        del state

        model.float().eval().to(device)
        prompt_ids = _prompt_ids(model)
        if prompt_ids != list(cfg.expected_prompt_ids):
            raise TranslatorError(
                f"the prompt tokenizes to {prompt_ids}, expected {list(cfg.expected_prompt_ids)}; "
                "the tokenizer does not match the checkpoint"
            )
        if model.lang != "English":
            raise TranslatorError(f"model prompt language is {model.lang!r}, expected English")
        report = LoadReport(
            checkpoint_name=name,
            checkpoint_sha256=ref.sha256,
            device=device,
            parameters=sum(p.numel() for p in model.parameters()),
            mt5_parameters=mt5_parameters,
            checkpoint_tensors=n_tensors,
            output_layer_was_joined=joined,
            seconds=time.perf_counter() - started,
        )
        return cls(model, cfg, device, report)

    # --- inputs ------------------------------------------------------------------------

    def _batch(self, model_inputs: ClipInputs | Sequence[ClipInputs]) -> tuple[dict[str, Any], int]:
        """The reference collate (`datasets.py:346-376`) for float32 clips: pad each clip to
        the longest by repeating its last frame, with an attention mask over real frames."""
        import torch

        clips = [model_inputs] if isinstance(model_inputs, Mapping) else list(model_inputs)
        if not clips:
            raise TranslatorError("no clips given")
        lengths = []
        for clip in clips:
            if set(clip) != set(PARTS):
                raise TranslatorError(
                    f"a clip needs exactly the groups {PARTS}, got {sorted(clip)}"
                )
            t = clip["body"].shape[0]
            for part in PARTS:
                arr = clip[part]
                if arr.shape != (t, PART_SIZES[part], 3) or arr.dtype != np.float32:
                    raise TranslatorError(
                        f"group {part!r}: expected float32 [{t}, {PART_SIZES[part]}, 3], "
                        f"got {arr.dtype} {arr.shape}"
                    )
            if t == 0:
                raise TranslatorError("a clip has no frames")
            if t > self.cfg.max_frames:
                raise TranslatorError(
                    f"a clip has {t} frames; the base model's maximum is {self.cfg.max_frames}. "
                    "Thin it first (infer.offline.select_frames)."
                )
            lengths.append(t)
        longest = max(lengths)
        src: dict[str, Any] = {}
        for part in PARTS:
            rows = []
            for clip, t in zip(clips, lengths, strict=True):
                x = torch.from_numpy(np.ascontiguousarray(clip[part]))
                if t < longest:
                    x = torch.cat([x, x[-1:].expand(longest - t, -1, -1)], dim=0)
                rows.append(x)
            src[part] = torch.stack(rows).to(self.device)
        mask = torch.zeros(len(clips), longest, dtype=torch.long)
        for i, t in enumerate(lengths):
            mask[i, :t] = 1
        src["attention_mask"] = mask.to(self.device)
        return src, len(clips)

    def _embed(self, src: dict[str, Any], n: int) -> dict[str, Any]:
        """Run the reference `forward` up to the point where it calls mT5, and take what it
        was about to pass: prompt plus pose embeddings, and the attention mask."""
        captured: dict[str, Any] = {}

        def stop(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            captured["inputs_embeds"] = kwargs["inputs_embeds"]
            captured["attention_mask"] = kwargs["attention_mask"]
            raise _StopAtMt5

        handle = self._model.mt5_model.register_forward_pre_hook(stop, with_kwargs=True)
        try:
            self._model(src, {"gt_sentence": [""] * n, "gt_gloss": [""] * n})
        except _StopAtMt5:
            pass
        else:
            raise TranslatorError("the reference forward did not reach mT5 as expected")
        finally:
            handle.remove()
        return captured

    # --- the interface -----------------------------------------------------------------

    def translate(
        self,
        model_inputs: ClipInputs | Sequence[ClipInputs],
        decode_cfg: DecodeConfig | None = None,
    ) -> list[str]:
        """English text for one clip or a batch of clips."""
        return [text for text, _ in self.translate_with_ids(model_inputs, decode_cfg)]

    def translate_with_ids(
        self,
        model_inputs: ClipInputs | Sequence[ClipInputs],
        decode_cfg: DecodeConfig | None = None,
    ) -> list[tuple[str, list[int]]]:
        """As `translate`, also returning the output token ids (for parity tests).

        Clips are decoded one at a time, never padded against each other. The base model's
        padding (repeat the last frame) reaches back through its temporal convolutions and
        changes the features of a shorter clip's last six frames, which is enough to change
        the text; decoding singly makes the result independent of what else is in the list.
        """
        import torch

        decode = decode_cfg or self.cfg.decode
        clips = [model_inputs] if isinstance(model_inputs, Mapping) else list(model_inputs)
        self._model.eval()
        out: list[tuple[str, list[int]]] = []
        with torch.no_grad():
            for clip in clips:
                src, n = self._batch(clip)
                embedded = self._embed(src, n)
                ids = self._model.generate(
                    embedded, max_new_tokens=decode.max_new_tokens, num_beams=decode.num_beams
                )
                text = self._model.mt5_tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
                out.append((str(text), [int(i) for i in ids[0].tolist()]))
            self._sync()
        return out

    def embed_for_test(self, model_inputs: ClipInputs | Sequence[ClipInputs]) -> tuple[Any, Any]:
        """`(inputs_embeds, attention_mask)` as the reference would hand them to mT5, with the
        reference's batch padding. For tests that measure what padding does."""
        import torch

        self._model.eval()
        with torch.no_grad():
            src, n = self._batch(model_inputs)
            embedded = self._embed(src, n)
        return embedded["inputs_embeds"], embedded["attention_mask"]

    def translate_reference_path(
        self, model_inputs: ClipInputs, decode_cfg: DecodeConfig | None = None
    ) -> tuple[str, list[int]]:
        """The reference's own two-step path, extra mT5 pass included
        (`demo/online_inference.py:123-128`). Kept so a test can show `translate` equals it."""
        import torch

        decode = decode_cfg or self.cfg.decode
        self._model.eval()
        with torch.no_grad():
            src, n = self._batch(model_inputs)
            stack = self._model(src, {"gt_sentence": [""] * n, "gt_gloss": [""] * n})
            ids = self._model.generate(
                stack, max_new_tokens=decode.max_new_tokens, num_beams=decode.num_beams
            )
        text = self._model.mt5_tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        return str(text), ids[0].tolist()

    def forward(
        self, model_inputs: ClipInputs | Sequence[ClipInputs], labels: Sequence[str]
    ) -> Any:
        """The reference model's own training loss for these clips and English sentences:
        `Uni_Sign.forward(...)['loss']`, unchanged (cross-entropy with the configured label
        smoothing, targets cut at 50 tokens; `models.py:282-301`). Returns a scalar tensor.

        Gradients, train or eval mode, and everything else about training are the caller's
        business; this module defines no loss of its own and no training code.
        """
        src, n = self._batch(model_inputs)
        if len(labels) != n:
            raise TranslatorError(f"{n} clips but {len(labels)} label sentences")
        return self._model(src, {"gt_sentence": list(labels), "gt_gloss": [""] * n})["loss"]

    # --- small accessors ---------------------------------------------------------------

    @property
    def torch_module(self) -> Any:
        """The underlying `torch.nn.Module`, for code that manages parameters (adapters,
        training). Its definition belongs to the reference code."""
        return self._model

    def prompt_ids(self) -> list[int]:
        return _prompt_ids(self._model)

    def _sync(self) -> None:
        if self.device == "mps":
            import torch

            torch.mps.synchronize()


def _prompt_ids(model: Any) -> list[int]:
    prompt = f"Translate sign language video to {model.lang}: "
    encoded = model.mt5_tokenizer([prompt], padding="longest", truncation=True, return_tensors="pt")
    return [int(i) for i in encoded["input_ids"][0].tolist()]
