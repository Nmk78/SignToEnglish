"""Canonical-JSON hashing for configs, and SHA-256 for files.

Config hash (ARCHITECTURE.md, section 3): SHA-256 of the canonical JSON of a config
model, truncated to 16 hex characters. Paths never enter a hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath
from typing import Any

from pydantic import BaseModel

CONFIG_HASH_LENGTH = 16
_CHUNK = 1024 * 1024


class HashMismatchError(RuntimeError):
    """A file's content hash differs from the pinned one."""


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, ASCII only, no NaN or infinity.

    Accepts JSON-native values only (dict with string keys, list, tuple, str, int, float,
    bool, None). Anything else raises `TypeError`, so nothing is hashed by its `repr`.
    """
    _check_json_native(obj, "$")
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _check_json_native(obj: Any, where: str) -> None:
    if obj is None or isinstance(obj, str | bool | int | float):
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string key {key!r} at {where}")
            _check_json_native(value, f"{where}.{key}")
        return
    if isinstance(obj, list | tuple):
        for i, value in enumerate(obj):
            _check_json_native(value, f"{where}[{i}]")
        return
    raise TypeError(f"value of type {type(obj).__name__} at {where} is not JSON-native")


def hash_obj(obj: Any, length: int = CONFIG_HASH_LENGTH) -> str:
    """SHA-256 of `canonical_json(obj)`, truncated to `length` hex characters."""
    digest = hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()
    return digest[:length]


def _reject_paths(obj: Any, where: str) -> None:
    if isinstance(obj, PurePath):
        raise TypeError(
            f"config field {where} is a path; paths are machine-specific and must not be hashed"
        )
    if isinstance(obj, dict):
        for key, value in obj.items():
            _reject_paths(value, f"{where}.{key}")
    elif isinstance(obj, list | tuple | set | frozenset):
        for i, value in enumerate(obj):
            _reject_paths(value, f"{where}[{i}]")


def config_hash(cfg: BaseModel, length: int = CONFIG_HASH_LENGTH) -> str:
    """Hash of a config model. Refuses a model that contains a path-typed value."""
    _reject_paths(cfg.model_dump(mode="python"), type(cfg).__name__)
    return hash_obj(cfg.model_dump(mode="json"), length)


def sha256_file(path: str | Path) -> str:
    """Full SHA-256 hex digest of a file's content, read in chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise `HashMismatchError` unless the file's SHA-256 equals `expected` (64 hex chars)."""
    expected = expected.strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"expected SHA-256 must be 64 hex characters, got {expected!r}")
    actual = sha256_file(path)
    if actual != expected:
        raise HashMismatchError(f"{path}: SHA-256 is {actual}, expected {expected}")
