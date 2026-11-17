"""Offline translation: one sentence clip in, one English sentence out.

    video file ─► cache or extraction ─► (time slice) ─► thin to the model's maximum
               ─► normalize_keypoints ─► Translator.translate ─► text

Everything before the model is the shared preprocessing path; nothing is re-implemented
here. Output per clip (ARCHITECTURE.md, section 4.6): `{"id", "text", "n_frames",
"decode_ms"}`, plus a few fields that say what was done.

Inputs longer than the base model's maximum (256 frames) are thinned to evenly spaced
frames, first and last included. The reference picked a random subset instead, at inference
as well as in training; an evenly spaced subset is one of the subsets it could have drawn,
and it makes the result repeatable. Thinning happens on the raw keypoints, before
normalization, as in the reference, because the body box is computed over the frames used.

This module can also read keypoints in the format the Uni-Sign authors publish (one pickle
per clip, coordinates already divided by frame width and height), which is what the
How2Sign reproduction check uses. Those pickles and label files are downloaded data, so
they are read with an unpickler that accepts plain containers and NumPy arrays only.
"""

from __future__ import annotations

import gzip
import io
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from signtoenglish.common.config import PathsConfig
from signtoenglish.infer.loader import LoadedModel
from signtoenglish.models.translator import DecodeConfig
from signtoenglish.preprocess.cache import KeypointCache, Source, get_or_extract
from signtoenglish.preprocess.extract import PoseCore, extract_file
from signtoenglish.preprocess.normalize import normalize_keypoints
from signtoenglish.preprocess.schema import NUM_KEYPOINTS, RawKeypoints


class OfflineError(RuntimeError):
    """A clip could not be translated; the message says why."""


def select_frames(n_frames: int, max_frames: int) -> npt.NDArray[np.int64]:
    """Indices of the frames to use: all of them up to `max_frames`, otherwise `max_frames`
    evenly spaced indices from the first frame to the last. Deterministic and increasing."""
    if n_frames < 0 or max_frames < 1:
        raise ValueError(f"bad arguments: n_frames={n_frames}, max_frames={max_frames}")
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=np.int64)
    return np.round(np.linspace(0, n_frames - 1, max_frames)).astype(np.int64)


@dataclass(frozen=True, slots=True)
class Translation:
    id: str
    text: str
    n_frames: int  # frames available in the clip (after any time slice)
    n_frames_used: int  # frames given to the model (at most the model's maximum)
    decode_ms: float
    token_ids: tuple[int, ...]
    keypoints_from: str  # "cache", "extracted", "direct", or "published"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "n_frames": self.n_frames,
            "decode_ms": round(self.decode_ms, 1),
            "n_frames_used": self.n_frames_used,
            "keypoints_from": self.keypoints_from,
        }


class OfflineTranslator:
    """Translates sentence clips with a loaded model. Pose models load on first need."""

    def __init__(self, loaded: LoadedModel, paths: PathsConfig, use_cache: bool = True) -> None:
        self.loaded = loaded
        self._paths = paths
        self._cache = KeypointCache(paths.cache_root, loaded.keypoint_cfg) if use_cache else None
        self._core: PoseCore | None = None

    def _pose_core(self) -> PoseCore:
        if self._core is None:
            self._core = PoseCore(self.loaded.keypoint_cfg, self._paths.assets_root)
        return self._core

    # --- from arrays -------------------------------------------------------------------

    def translate_arrays(
        self,
        keypoints: npt.NDArray[np.floating],
        scores: npt.NDArray[np.floating],
        frame_size: tuple[int, int],
        *,
        clip_id: str = "clip",
        keypoints_from: str = "direct",
        decode: DecodeConfig | None = None,
    ) -> Translation:
        """Raw keypoints `[T, 133, 2]` (in the units of `frame_size`) and scores `[T, 133]`."""
        n = int(keypoints.shape[0])
        if n == 0:
            raise OfflineError(f"{clip_id}: no frames to translate")
        translator = self.loaded.translator
        keep = select_frames(n, translator.cfg.max_frames)
        inputs, _ = normalize_keypoints(
            keypoints[keep],
            scores[keep],
            np.ones(len(keep), dtype=np.bool_),
            frame_size,
            self.loaded.model_cfg.normalize,
        )
        started = time.perf_counter()
        clip = {str(part): array for part, array in inputs.items()}
        text, ids = translator.translate_with_ids(clip, decode)[0]
        return Translation(
            id=clip_id,
            text=text,
            n_frames=n,
            n_frames_used=len(keep),
            decode_ms=1000 * (time.perf_counter() - started),
            token_ids=tuple(ids),
            keypoints_from=keypoints_from,
        )

    def translate_raw(
        self,
        raw: RawKeypoints,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        clip_id: str = "clip",
        keypoints_from: str = "direct",
        decode: DecodeConfig | None = None,
    ) -> Translation:
        """Cached or freshly extracted keypoints, optionally cut to `[start_ms, end_ms)` in
        the file's own timestamps."""
        keep = np.ones(len(raw), dtype=np.bool_)
        if start_ms is not None:
            keep &= raw.timestamps_ms >= start_ms
        if end_ms is not None:
            keep &= raw.timestamps_ms < end_ms
        if not keep.any():
            raise OfflineError(
                f"{clip_id}: no frames between {start_ms} and {end_ms} ms "
                f"(the file runs from {int(raw.timestamps_ms[0]) if len(raw) else None} to "
                f"{int(raw.timestamps_ms[-1]) if len(raw) else None} ms)"
            )
        return self.translate_arrays(
            raw.keypoints[keep],
            raw.scores[keep],
            raw.frame_size,
            clip_id=clip_id,
            keypoints_from=keypoints_from,
            decode=decode,
        )

    # --- from a video file -------------------------------------------------------------

    def translate_file(
        self,
        path: str | Path,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        expected_sha256: str | None = None,
        clip_id: str | None = None,
        decode: DecodeConfig | None = None,
    ) -> Translation:
        """One sentence from a video file. Keypoints come from the cache when it has them."""
        path = Path(path)
        clip_id = clip_id or path.name
        if self._cache is None:
            if not path.is_file():
                raise OfflineError(f"{clip_id}: source file not found: {path}")
            raw, origin = extract_file(path, self._pose_core()), "extracted"
        else:
            source = Source(path=path, expected_sha256=expected_sha256, origin="offline")
            entry, outcome = get_or_extract(source, self._cache, self._pose_core)
            if entry is None:
                raise OfflineError(f"{clip_id}: {outcome.detail}")
            raw, origin = entry.raw, outcome.status  # "cached" or "extracted"
            origin = "cache" if origin == "cached" else origin
        return self.translate_raw(
            raw,
            start_ms=start_ms,
            end_ms=end_ms,
            clip_id=clip_id,
            keypoints_from=origin,
            decode=decode,
        )

    # --- from keypoints in the Uni-Sign authors' published format ---------------------------

    def translate_published(
        self,
        pose_pickle: str | Path,
        *,
        clip_id: str | None = None,
        decode: DecodeConfig | None = None,
    ) -> Translation:
        keypoints, scores = load_published_pose(pose_pickle)
        return self.translate_arrays(
            keypoints,
            scores,
            (1, 1),  # already divided by frame width and height
            clip_id=clip_id or Path(pose_pickle).stem,
            keypoints_from="published",
            decode=decode,
        )


# --- reading downloaded pickles safely -------------------------------------------------------

_SAFE_GLOBALS = {
    ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"), ("builtins", "set"),
    ("builtins", "str"), ("builtins", "int"), ("builtins", "float"), ("builtins", "bool"),
    ("builtins", "bytes"), ("builtins", "slice"), ("collections", "OrderedDict"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
}  # fmt: skip


class _DataOnlyUnpickler(pickle.Unpickler):
    """Unpickles plain containers and NumPy arrays; refuses every other class, so a
    downloaded file cannot run code by being loaded."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _SAFE_GLOBALS:
            return super().find_class(module, name)
        raise OfflineError(
            f"refusing to unpickle {module}.{name}: only plain containers and NumPy arrays "
            "are accepted from downloaded files"
        )


def _load_data_pickle(data: bytes, what: str) -> Any:
    try:
        return _DataOnlyUnpickler(io.BytesIO(data)).load()
    except OfflineError:
        raise
    except Exception as exc:
        raise OfflineError(f"cannot read {what}: {exc}") from exc


def load_published_pose(
    path: str | Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """One clip in the Uni-Sign authors' pose format (`demo/pose_extraction.py:23-49`,
    `datasets.py:461-488`): `{"keypoints": T x (1, 133, 2), "scores": T x (1, 133)}`, with an
    optional `start` and `end` frame range. Returns `[T, 133, 2]` and `[T, 133]`; coordinates
    are fractions of the frame width and height."""
    path = Path(path)
    if not path.is_file():
        raise OfflineError(f"pose file not found: {path}")
    pose = _load_data_pickle(path.read_bytes(), str(path))
    if not isinstance(pose, dict) or "keypoints" not in pose or "scores" not in pose:
        raise OfflineError(f"{path}: expected a dict with 'keypoints' and 'scores'")
    first, last = 0, len(pose["scores"])
    if "start" in pose:
        first, last = int(pose["start"]), int(pose["end"])
        if not 0 <= first < last <= len(pose["scores"]):
            raise OfflineError(f"{path}: bad frame range {first}..{last}")
    try:
        # The reference keeps the first person of each frame (`datasets.py:28-29`).
        keypoints = np.stack(
            [np.asarray(k, dtype=np.float64)[0] for k in pose["keypoints"][first:last]]
        )
        scores = np.stack([np.asarray(s, dtype=np.float64)[0] for s in pose["scores"][first:last]])
    except (ValueError, IndexError, TypeError) as exc:
        raise OfflineError(f"{path}: keypoints are not in the expected layout: {exc}") from exc
    if keypoints.shape[1:] != (NUM_KEYPOINTS, 2) or scores.shape != keypoints.shape[:2]:
        raise OfflineError(f"{path}: unexpected shapes {keypoints.shape} and {scores.shape}")
    return keypoints, scores


@dataclass(frozen=True, slots=True)
class PublishedLabel:
    name: str
    video_path: str
    text: str


def load_published_labels(path: str | Path) -> list[PublishedLabel]:
    """A Uni-Sign label file (`utils.py:315-318`, `datasets.py:446-457`): a gzip-compressed
    pickle of `{key: {"name", "video_path", "text", ...}}`, returned in file order."""
    path = Path(path)
    if not path.is_file():
        raise OfflineError(f"label file not found: {path}")
    try:
        data = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as exc:
        raise OfflineError(f"{path} is not a gzip file: {exc}") from exc
    raw = _load_data_pickle(data, str(path))
    if not isinstance(raw, dict):
        raise OfflineError(f"{path}: expected a dict of samples")
    labels = []
    for key, sample in raw.items():
        if not isinstance(sample, dict) or not all(
            isinstance(sample.get(k), str) for k in ("name", "video_path", "text")
        ):
            raise OfflineError(f"{path}: sample {key!r} lacks string fields name, video_path, text")
        labels.append(PublishedLabel(sample["name"], sample["video_path"], sample["text"]))
    return labels


def published_pose_path(pose_dir: Path, label: PublishedLabel) -> Path:
    """Where the reference looks for a sample's pose file (`datasets.py:462`)."""
    return pose_dir / label.video_path.replace(".mp4", ".pkl")
