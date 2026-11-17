"""Fixtures for this folder come from tests/models/shared_fixtures.py (one copy per process)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[1] / "models" / "shared_fixtures.py"
_NAME = "s2e_shared_model_fixtures"
if _NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_NAME, _SHARED)
    assert _spec is not None and _spec.loader is not None
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _module
    _spec.loader.exec_module(_module)
for _fixture in sys.modules[_NAME].FIXTURE_NAMES:
    globals()[_fixture] = getattr(sys.modules[_NAME], _fixture)
