"""Keypoint cache on disk (ARCHITECTURE.md, sections 3 and 4.4).

One `.npz` per source video, addressed by what determines its content and nothing else:

    <cache_root>/keypoints/<preprocessing_hash>/<source_sha256[:2]>/<source_sha256>.npz

- `preprocessing_hash` covers every extraction setting, the SHA-256 of each pose model
  file, the versions of the libraries that produce the numbers, the execution providers,
  and `EXTRACTOR_VERSION`. Paths are not in it, so moving the assets or the cache does not
  invalidate anything.
- `source_sha256` is the content hash of the video file, so a copied, renamed, or
  re-timestamped file hits the same entry.

Values are raw estimator output; normalization happens at load time elsewhere and never
invalidates the cache. An entry is written to a temporary name and renamed, and is read back
strictly: anything that is not a complete entry for exactly this hash and source reads as
absent, never as data.

This module also reads the two manifest fields it needs (`source_path`, `source_sha256`)
straight from JSON lines, because `preprocess` may not import the `data` package.
"""

from __future__ import annotations

import contextlib
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from signtoenglish.common.hashing import hash_obj, sha256_file
from signtoenglish.preprocess.schema import (
    EXTRACTOR_VERSION,
    NUM_KEYPOINTS,
    KeypointConfig,
    RawKeypoints,
)
from signtoenglish.preprocess.video_io import ReadStats

CACHE_SCHEMA_VERSION = 1
_ARRAY_KEYS = ("timestamps_ms", "keypoints", "scores", "person_found")
#: Libraries whose version can change the extracted numbers. ARCHITECTURE.md names rtmlib
#: and onnxruntime; the decoder and the image library the estimator calls are included too.
_VERSIONED_PACKAGES = ("rtmlib", "onnxruntime", "av", "opencv-python", "numpy")


def execution_providers(cfg: KeypointConfig) -> dict[str, str]:
    """The provider each model is asked to run on. The detector is always on CPU."""
    return {"detector": "cpu", "pose": cfg.pose_provider}


def preprocessing_fingerprint(
    cfg: KeypointConfig, extractor_version: int = EXTRACTOR_VERSION
) -> dict[str, Any]:
    """Everything that determines extraction output, as plain JSON values.

    A model file contributes its SHA-256 and input size, not its location.
    """
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "extractor_version": extractor_version,
        "detector": {"sha256": cfg.detector.sha256, "input_size": list(cfg.detector.input_size)},
        "pose": {"sha256": cfg.pose.sha256, "input_size": list(cfg.pose.input_size)},
        "execution_providers": execution_providers(cfg),
        "model_rate_fps": cfg.model_rate_fps,
        "rate_slack": cfg.rate_slack,
        "person_rule": "largest_box_else_whole_frame",
        "versions": {name: version(name) for name in _VERSIONED_PACKAGES},
    }


def preprocessing_hash(cfg: KeypointConfig, extractor_version: int = EXTRACTOR_VERSION) -> str:
    """16 hex characters identifying the extraction that produced a cache directory."""
    return hash_obj(preprocessing_fingerprint(cfg, extractor_version))


@dataclass(frozen=True, slots=True)
class CacheEntry:
    raw: RawKeypoints
    meta: dict[str, Any]


class KeypointCache:
    """Read and write entries for one preprocessing configuration."""

    def __init__(self, cache_root: Path, cfg: KeypointConfig) -> None:
        self.cfg = cfg
        self.hash = preprocessing_hash(cfg)
        self.dir = Path(cache_root) / "keypoints" / self.hash

    def path_for(self, source_sha256: str) -> Path:
        sha = _check_sha(source_sha256)
        return self.dir / sha[:2] / f"{sha}.npz"

    def load(self, source_sha256: str) -> CacheEntry | None:
        """The entry, or None if there is no complete, matching entry."""
        entry, _ = self.inspect(source_sha256)
        return entry

    def inspect(self, source_sha256: str) -> tuple[CacheEntry | None, str]:
        """`(entry, "ok")`, `(None, "missing")`, or `(None, "corrupt: <why>")`."""
        path = self.path_for(source_sha256)
        if not path.is_file():
            return None, "missing"
        try:
            with np.load(path, allow_pickle=False) as data:
                missing = [k for k in (*_ARRAY_KEYS, "meta") if k not in data.files]
                if missing:
                    return None, f"corrupt: missing arrays {missing}"
                meta = json.loads(str(data["meta"][()]))
                arrays = {k: data[k] for k in _ARRAY_KEYS}
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
            return None, f"corrupt: unreadable ({type(exc).__name__}: {exc})"
        if not isinstance(meta, dict):
            return None, "corrupt: meta is not an object"
        if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None, f"corrupt: schema_version {meta.get('schema_version')!r}"
        if meta.get("preprocessing_hash") != self.hash:
            return (
                None,
                f"corrupt: written for preprocessing hash {meta.get('preprocessing_hash')!r}",
            )
        if meta.get("source_sha256") != source_sha256:
            return None, f"corrupt: written for source {meta.get('source_sha256')!r}"
        try:
            raw = RawKeypoints(
                timestamps_ms=arrays["timestamps_ms"],
                keypoints=arrays["keypoints"],
                scores=arrays["scores"],
                person_found=arrays["person_found"],
                width=int(meta["width"]),
                height=int(meta["height"]),
            )
        except (ValueError, KeyError, TypeError) as exc:
            return None, f"corrupt: {exc}"
        if meta.get("n_kept_frames") != len(raw):
            return None, f"corrupt: {len(raw)} frames, meta says {meta.get('n_kept_frames')!r}"
        return CacheEntry(raw=raw, meta=meta), "ok"

    def store(self, source_sha256: str, raw: RawKeypoints, stats: ReadStats) -> Path:
        """Write an entry atomically. A crash at any point leaves either the previous state
        or the complete new entry; the temporary file never has a name that is read."""
        path = self.path_for(source_sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_dir_meta()
        info = stats.info
        meta = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "width": raw.width,
            "height": raw.height,
            "nominal_fps": info.nominal_fps if info else None,
            "n_source_frames": stats.decoded,
            "n_kept_frames": len(raw),
            "n_dropped_duplicate_timestamps": stats.dropped_non_increasing,
            "n_dropped_without_timestamp": stats.dropped_without_timestamp,
            "model_rate_fps": self.cfg.model_rate_fps,
            "preprocessing_hash": self.hash,
            "num_keypoints": NUM_KEYPOINTS,
            "pose_models": {
                "detector": {
                    "name": Path(self.cfg.detector.relative_path).name,
                    "sha256": self.cfg.detector.sha256,
                },
                "pose": {
                    "name": Path(self.cfg.pose.relative_path).name,
                    "sha256": self.cfg.pose.sha256,
                },
            },
            "rtmlib_version": version("rtmlib"),
            "onnxruntime_version": version("onnxruntime"),
            "execution_provider": execution_providers(self.cfg),
            "extractor_version": EXTRACTOR_VERSION,
            "person_rule": "largest_box_else_whole_frame",
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
        try:
            with tmp.open("wb") as fh:
                np.savez(
                    fh,
                    timestamps_ms=raw.timestamps_ms,
                    keypoints=raw.keypoints,
                    scores=raw.scores,
                    person_found=raw.person_found,
                    meta=np.array(json.dumps(meta, sort_keys=True)),
                )
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
        return path

    def _write_dir_meta(self) -> None:
        """`meta.json` in the hash directory: what this hash stands for, in full."""
        target = self.dir / "meta.json"
        if target.is_file():
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        body = {
            "preprocessing_hash": self.hash,
            "fingerprint": preprocessing_fingerprint(self.cfg),
            "config": self.cfg.model_dump(mode="json"),
        }
        tmp = target.with_name(f".meta.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)


def _check_sha(value: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"expected a lowercase SHA-256 hex digest, got {value!r}")
    return value


# --- sources: files and manifest rows ----------------------------------------------------


class ManifestError(ValueError):
    """A manifest file cannot be read for the two fields the cache needs."""


@dataclass(frozen=True, slots=True)
class Source:
    """One source video to extract or verify. `expected_sha256` comes from a manifest row
    and is None for a file named directly."""

    path: Path
    expected_sha256: str | None
    origin: str  # "<manifest>:<line>" or "file"


def read_manifest_sources(manifest: Path, data_root: Path) -> list[Source]:
    """Sources named by a manifest (ARCHITECTURE.md, section 4.2), one per distinct video.

    Reads only `source_path` (relative to the data root) and `source_sha256`, which clip
    manifests and continuous-take manifests both carry; all other fields, including `split`,
    are ignored here. Full validation belongs to `data.manifest`.
    """
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {manifest}: {exc}") from exc
    seen: dict[str, Source] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{manifest}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{where}: not JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise ManifestError(f"{where}: row is not an object")
        rel, sha = row.get("source_path"), row.get("source_sha256")
        if not isinstance(rel, str) or not rel or not isinstance(sha, str):
            raise ManifestError(f"{where}: needs string fields source_path and source_sha256")
        try:
            _check_sha(sha)
        except ValueError as exc:
            raise ManifestError(f"{where}: {exc}") from exc
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ManifestError(f"{where}: source_path must be relative to the data root: {rel!r}")
        previous = seen.get(sha)
        if previous is not None and previous.path != data_root / rel:
            continue  # same content under another name: one cache entry serves both
        seen.setdefault(sha, Source(path=data_root / rel, expected_sha256=sha, origin=where))
    return list(seen.values())


@dataclass(frozen=True, slots=True)
class ExtractOutcome:
    source: Source
    status: str  # "cached", "extracted", or "failed"
    source_sha256: str | None
    frames: int
    seconds: float
    detail: str = ""


def get_or_extract(
    source: Source, cache: KeypointCache, core_factory: Any
) -> tuple[CacheEntry | None, ExtractOutcome]:
    """Return the cached keypoints for a source, extracting and storing them if needed.

    `core_factory()` returns the `PoseCore`; it is only called on a miss, so a fully cached
    run never loads the pose models. A failure on one source is reported, not raised, so a
    run over a list can continue and be resumed.
    """
    import time

    from signtoenglish.preprocess.extract import extract_file
    from signtoenglish.preprocess.video_io import VideoError

    start = time.perf_counter()

    def outcome(status: str, sha: str | None, frames: int = 0, detail: str = "") -> ExtractOutcome:
        return ExtractOutcome(source, status, sha, frames, time.perf_counter() - start, detail)

    if not source.path.is_file():
        return None, outcome("failed", None, detail=f"source file not found: {source.path}")
    sha = sha256_file(source.path)
    if source.expected_sha256 is not None and sha != source.expected_sha256:
        return None, outcome(
            "failed",
            sha,
            detail=f"content hash {sha} differs from the manifest's {source.expected_sha256}",
        )
    entry = cache.load(sha)
    if entry is not None:
        return entry, outcome("cached", sha, len(entry.raw))
    try:
        stats = ReadStats()
        raw = extract_file(source.path, core_factory(), stats)
    except VideoError as exc:
        return None, outcome("failed", sha, detail=str(exc))
    if len(raw) == 0:
        return None, outcome("failed", sha, detail="no frames with timestamps could be decoded")
    cache.store(sha, raw, stats)
    stored = cache.load(sha)
    if stored is None:
        return None, outcome("failed", sha, detail="entry did not read back after writing")
    return stored, outcome("extracted", sha, len(raw))


@dataclass(frozen=True, slots=True)
class VerifyResult:
    source: Source
    status: str  # "ok", "missing", "corrupt: ...", "source missing", "source changed"


def verify_sources(
    sources: list[Source], cache: KeypointCache, check_sources: bool = False
) -> list[VerifyResult]:
    """Check each manifest source against the cache. With `check_sources`, also re-hash the
    video files and report any that are gone or no longer match their manifest hash."""
    results = []
    for source in sources:
        sha = source.expected_sha256
        if sha is None:
            if not source.path.is_file():
                results.append(VerifyResult(source, "source missing"))
                continue
            sha = sha256_file(source.path)
        elif check_sources:
            if not source.path.is_file():
                results.append(VerifyResult(source, "source missing"))
                continue
            if sha256_file(source.path) != sha:
                results.append(VerifyResult(source, "source changed"))
                continue
        _, status = cache.inspect(sha)
        results.append(VerifyResult(source, status))
    return results
