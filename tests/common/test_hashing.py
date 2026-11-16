from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from signtoenglish.common.config import FrozenConfig, PathsConfig, load_config
from signtoenglish.common.hashing import (
    HashMismatchError,
    canonical_json,
    config_hash,
    hash_obj,
    sha256_file,
    verify_sha256,
)


class Inner(FrozenConfig):
    rate: float = 1.0
    names: tuple[str, ...] = ()


class Example(FrozenConfig):
    name: str
    hop: int = 3
    inner: Inner = Inner()


def test_canonical_json_is_exact() -> None:
    obj = {"b": [1, 2.5, None, True], "a": {"z": "é", "y": (1, 2)}}
    assert canonical_json(obj) == '{"a":{"y":[1,2],"z":"\\u00e9"},"b":[1,2.5,null,true]}'


def test_canonical_json_ignores_key_order() -> None:
    assert canonical_json({"a": 1, "b": {"c": 2, "d": 3}}) == canonical_json(
        {"b": {"d": 3, "c": 2}, "a": 1}
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_numbers(bad: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": bad})


@pytest.mark.parametrize("bad", [{1: "a"}, {"p": Path("x")}, {"s": {1, 2}}, {"b": b"x"}, object()])
def test_canonical_json_rejects_non_json_values(bad: object) -> None:
    with pytest.raises(TypeError):
        canonical_json(bad)


def test_hash_obj_matches_a_hand_computed_value() -> None:
    expected = hashlib.sha256(b'{"a":1,"b":"x"}').hexdigest()
    assert hash_obj({"b": "x", "a": 1}) == expected[:16]
    assert hash_obj({"b": "x", "a": 1}, length=64) == expected


def test_config_hash_is_16_hex_and_known() -> None:
    cfg = Example(name="a")
    expected = hashlib.sha256(b'{"hop":3,"inner":{"names":[],"rate":1.0},"name":"a"}').hexdigest()[
        :16
    ]
    assert config_hash(cfg) == expected
    assert len(config_hash(cfg)) == 16


def test_config_hash_changes_with_any_field_and_only_with_content() -> None:
    base = Example(name="a")
    assert config_hash(base) == config_hash(Example(name="a", hop=3, inner=Inner(rate=1.0)))
    assert config_hash(base) != config_hash(Example(name="b"))
    assert config_hash(base) != config_hash(Example(name="a", hop=4))
    assert config_hash(base) != config_hash(Example(name="a", inner=Inner(rate=1.5)))
    assert config_hash(base) != config_hash(Example(name="a", inner=Inner(names=("x",))))


def test_config_hash_ignores_yaml_layout(tmp_path: Path) -> None:
    one = tmp_path / "one.yaml"
    two = tmp_path / "two.yaml"
    one.write_text("name: a\nhop: 5\ninner: {rate: 2.0}\n")
    two.write_text("# comment\ninner:\n  rate: 2.0\nhop: 5\nname: 'a'\n")
    assert config_hash(load_config(one, Example)) == config_hash(load_config(two, Example))


def test_config_hash_is_stable_across_processes(repo_root: Path) -> None:
    """Hashes are persisted in checkpoints and cache paths, so they must not depend on
    per-process state such as string hash randomization."""
    import subprocess
    import sys

    code = (
        "from signtoenglish.common.hashing import hash_obj;"
        "print(hash_obj({'b': [1, 2.5], 'a': {'k': 'v'}, 'c': None}))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert outputs == {hash_obj({"b": [1, 2.5], "a": {"k": "v"}, "c": None})}


def test_paths_never_enter_a_hash(tmp_path: Path) -> None:
    class WithPath(FrozenConfig):
        model_file: Path

    with pytest.raises(TypeError, match="path"):
        config_hash(WithPath(model_file=tmp_path / "m.onnx"))
    with pytest.raises(TypeError, match="path"):
        config_hash(
            PathsConfig(
                data_root=tmp_path,
                cache_root=tmp_path,
                runs_root=tmp_path,
                assets_root=tmp_path,
                third_party_root=tmp_path,
            )
        )


def test_sha256_file_and_verify(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 9000  # larger than one read chunk
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    assert sha256_file(path) == digest
    verify_sha256(path, digest)
    verify_sha256(path, digest.upper())

    path.write_bytes(payload[:-1] + b"\x00")
    with pytest.raises(HashMismatchError, match=r"blob\.bin"):
        verify_sha256(path, digest)
    with pytest.raises(ValueError):
        verify_sha256(path, digest[:16])
