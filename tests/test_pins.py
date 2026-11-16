"""Pinned-version policy: every Python dependency is an exact `==` pin."""

import re
import tomllib
from pathlib import Path

import pytest

# name, optional extras, `==`, a concrete version (no wildcard), optional environment marker.
EXACT_PIN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,\s-]+\])?==[A-Za-z0-9][A-Za-z0-9.!+_-]*(\s*;.+)?$"
)


def unpinned(requirements: list[str]) -> list[str]:
    return [r for r in requirements if not EXACT_PIN.match(r.strip()) or "*" in r.split(";")[0]]


def all_requirements(pyproject: dict) -> dict[str, list[str]]:
    project = pyproject.get("project", {})
    groups: dict[str, list[str]] = {
        "project.dependencies": project.get("dependencies", []),
        "build-system.requires": pyproject.get("build-system", {}).get("requires", []),
    }
    for name, reqs in project.get("optional-dependencies", {}).items():
        groups[f"project.optional-dependencies.{name}"] = reqs
    for name, reqs in pyproject.get("dependency-groups", {}).items():
        groups[f"dependency-groups.{name}"] = reqs
    return groups


@pytest.mark.parametrize(
    "requirement",
    [
        "torch>=2.0",
        "torch~=2.15.0",
        "torch",
        "torch==2.*",
        "torch==2.15.0,<3",
        "torch>=2.0,==2.15.0",
        "torch @ https://example.invalid/torch.whl",
        "torch<3",
    ],
)
def test_checker_rejects_loose_requirements(requirement: str) -> None:
    assert unpinned([requirement]) == [requirement]


@pytest.mark.parametrize(
    "requirement",
    ["torch==2.15.0", "uvicorn[standard]==0.54.0", "x==1.0 ; sys_platform == 'darwin'"],
)
def test_checker_accepts_exact_pins(requirement: str) -> None:
    assert unpinned([requirement]) == []


def test_every_dependency_is_an_exact_pin(repo_root: Path) -> None:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    groups = all_requirements(pyproject)
    assert groups["project.dependencies"], "no dependencies found; is the parser wrong?"
    for group, reqs in groups.items():
        for req in reqs:
            assert not isinstance(req, dict), f"{group}: include-group tables are not allowed"
        assert unpinned(reqs) == [], f"{group} has entries that are not exact pins"


def test_python_is_one_pinned_minor_version(repo_root: Path) -> None:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    requires = pyproject["project"]["requires-python"]
    assert re.fullmatch(r"==3\.\d+\.\*", requires), requires
    minor = requires.removeprefix("==").removesuffix(".*")
    assert (repo_root / ".python-version").read_text().strip() == minor


def test_lockfile_is_committed_and_names_the_pins(repo_root: Path) -> None:
    lock = tomllib.loads((repo_root / "uv.lock").read_text())
    locked = {p["name"].lower().replace("_", "-"): p.get("version") for p in lock["package"]}
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    for group, reqs in all_requirements(pyproject).items():
        if group == "build-system.requires":
            continue  # build requirements are not part of uv.lock
        for req in reqs:
            name, version = re.split(r"==", req.split(";")[0].strip())
            name = re.sub(r"\[.*\]", "", name).lower().replace("_", "-")
            got = locked.get(name)
            assert got == version, f"{name}: pyproject {version}, lock {got}"


def test_running_interpreter_matches_the_pin(repo_root: Path) -> None:
    import sys

    minor = (repo_root / ".python-version").read_text().strip()
    assert f"{sys.version_info.major}.{sys.version_info.minor}" == minor
