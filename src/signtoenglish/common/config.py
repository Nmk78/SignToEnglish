"""Config loading: YAML files validated into frozen pydantic models.

Rules (ARCHITECTURE.md, section 3):
- every config is a `FrozenConfig`; unknown keys are an error;
- machine-specific paths live in `PathsConfig`, which is the one place environment
  variables are read, and paths never enter a config hash (see `hashing.config_hash`).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

PATHS_LOCAL_FILE = Path("configs") / "paths.local.yaml"
ENV_PREFIX = "S2E_"


class ConfigError(ValueError):
    """A config file is missing, malformed, or fails validation."""


class FrozenConfig(BaseModel):
    """Base class for every config model: immutable, and unknown keys are rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping. An empty file is an empty mapping."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)  # a SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict) or not all(isinstance(k, str) for k in data):
        raise ConfigError(f"{path}: top level must be a mapping with string keys")
    return data


def load_config[T: BaseModel](path: str | Path, model_cls: type[T]) -> T:
    """Load `path` and validate it as `model_cls`. Raises `ConfigError` with the file name."""
    data = load_yaml(path)
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def dump_config(cfg: BaseModel, path: str | Path) -> None:
    """Write the resolved config as YAML (for `runs/<id>/config.resolved.yaml`)."""
    path = Path(path)
    text = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=True, allow_unicode=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def find_repo_root(start: str | Path | None = None) -> Path:
    """Nearest directory at or above `start` (default: cwd) that holds `pyproject.toml`."""
    here = Path(start) if start is not None else Path.cwd()
    here = here.resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(f"no pyproject.toml found at or above {here}")


class PathsConfig(FrozenConfig):
    """Machine-specific locations. All are git-ignored directories; none enters a hash."""

    data_root: Path
    cache_root: Path
    runs_root: Path
    assets_root: Path
    third_party_root: Path


def load_paths(
    repo_root: str | Path | None = None, env: Mapping[str, str] | None = None
) -> PathsConfig:
    """Resolve paths: defaults under the repo root, then `configs/paths.local.yaml`,
    then environment variables `S2E_<FIELD>` (for example `S2E_DATA_ROOT`).

    Relative values are taken relative to the repo root. This function is the only place
    in the project that reads environment variables for settings.
    """
    import os

    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    environ = os.environ if env is None else env

    values: dict[str, Any] = {
        "data_root": "data",
        "cache_root": "cache",
        "runs_root": "runs",
        "assets_root": "assets",
        "third_party_root": "third_party",
    }
    local = root / PATHS_LOCAL_FILE
    if local.is_file():
        overrides = load_yaml(local)
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ConfigError(f"{local}: unknown keys {unknown}")
        values.update(overrides)
    for field in list(values):
        from_env = environ.get(ENV_PREFIX + field.upper())
        if from_env:
            values[field] = from_env

    resolved = {}
    for field, value in values.items():
        if not isinstance(value, str | Path):
            raise ConfigError(f"path setting {field} must be a string, got {value!r}")
        p = Path(value).expanduser()
        resolved[field] = p if p.is_absolute() else root / p
    return PathsConfig.model_validate(resolved)
