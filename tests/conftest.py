"""Shared test setup. Model hubs are offline in every test (CLAUDE.md, stop point 2)."""

import os
from pathlib import Path

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


# Tests that load the translation model carry the `model` marker so that `make check` stays
# quick; `make check-full` runs them. They are marked here by location, so tracks do not have
# to remember to do it.
_MODEL_TEST_PATHS = ("tests/models/", "tests/infer/", "tests/preprocess/test_parity.py")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        rel = Path(str(item.fspath)).resolve().relative_to(REPO_ROOT).as_posix()
        if rel.startswith(_MODEL_TEST_PATHS):
            item.add_marker(pytest.mark.model)
