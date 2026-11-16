from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from signtoenglish.common.config import (
    ConfigError,
    FrozenConfig,
    PathsConfig,
    dump_config,
    find_repo_root,
    load_config,
    load_paths,
    load_yaml,
)


class Inner(FrozenConfig):
    rate: float = 1.0


class Example(FrozenConfig):
    name: str
    hop: int = 3
    inner: Inner = Inner()


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_and_validates(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path / "c.yaml", "name: a\ninner:\n  rate: 2.5\n"), Example)
    assert cfg == Example(name="a", hop=3, inner=Inner(rate=2.5))


def test_models_are_frozen(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path / "c.yaml", "name: a\n"), Example)
    with pytest.raises(ValidationError):
        cfg.hop = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "text",
    [
        "name: a\nextra: 1\n",  # unknown key at the top level
        "name: a\ninner:\n  rate: 1\n  typo: 2\n",  # unknown key in a nested model
        "hop: 3\n",  # missing required field
        "name: a\nhop: not-a-number\n",  # wrong type
    ],
)
def test_invalid_content_is_a_config_error_naming_the_file(tmp_path: Path, text: str) -> None:
    path = write(tmp_path / "bad.yaml", text)
    with pytest.raises(ConfigError, match=r"bad\.yaml"):
        load_config(path, Example)


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="duplicate key"):
        load_yaml(write(tmp_path / "dup.yaml", "name: a\nname: b\n"))
    with pytest.raises(ConfigError, match="duplicate key"):
        load_yaml(write(tmp_path / "dup2.yaml", "inner:\n  rate: 1\n  rate: 2\n"))


def test_non_mapping_and_missing_files_are_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(write(tmp_path / "list.yaml", "- 1\n- 2\n"))
    with pytest.raises(ConfigError, match="cannot read"):
        load_yaml(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_yaml(write(tmp_path / "broken.yaml", "name: [unclosed\n"))
    assert load_yaml(write(tmp_path / "empty.yaml", "")) == {}


def test_yaml_cannot_construct_python_objects(tmp_path: Path) -> None:
    path = write(tmp_path / "evil.yaml", "name: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_yaml(path)


def test_dump_then_load_round_trips(tmp_path: Path) -> None:
    cfg = Example(name="résumé", hop=9, inner=Inner(rate=0.1))
    dump_config(cfg, tmp_path / "out.yaml")
    assert load_config(tmp_path / "out.yaml", Example) == cfg
    assert not (tmp_path / "out.yaml.tmp").exists()


def test_find_repo_root(repo_root: Path, tmp_path: Path) -> None:
    assert find_repo_root(repo_root / "src" / "signtoenglish" / "common") == repo_root
    with pytest.raises(ConfigError):
        find_repo_root(tmp_path)


def test_paths_default_to_directories_under_the_repo_root(tmp_path: Path) -> None:
    paths = load_paths(tmp_path, env={})
    root = tmp_path.resolve()
    assert paths == PathsConfig(
        data_root=root / "data",
        cache_root=root / "cache",
        runs_root=root / "runs",
        assets_root=root / "assets",
        third_party_root=root / "third_party",
    )


def test_paths_precedence_is_defaults_then_local_file_then_environment(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    write(tmp_path / "configs" / "paths.local.yaml", "data_root: /mnt/d\ncache_root: kp\n")
    paths = load_paths(tmp_path, env={"S2E_CACHE_ROOT": "/fast/cache", "UNRELATED": "x"})
    assert paths.data_root == Path("/mnt/d")
    assert paths.cache_root == Path("/fast/cache")
    assert paths.runs_root == tmp_path.resolve() / "runs"


def test_paths_local_file_rejects_unknown_keys(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    write(tmp_path / "configs" / "paths.local.yaml", "dataroot: /mnt/d\n")
    with pytest.raises(ConfigError, match="dataroot"):
        load_paths(tmp_path, env={})


def test_paths_example_file_matches_the_defaults(repo_root: Path, tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    example = (repo_root / "configs" / "paths.example.yaml").read_text()
    write(tmp_path / "configs" / "paths.local.yaml", example)
    with_example = load_paths(tmp_path, env={})
    (tmp_path / "configs" / "paths.local.yaml").unlink()
    assert with_example == load_paths(tmp_path, env={})
