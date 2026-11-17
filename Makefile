# One entry point for every check. Uses the locked environment only; never the system Python.
# Model hubs are offline by default (CLAUDE.md, stop point 2).
export HF_HUB_OFFLINE := 1
export TRANSFORMERS_OFFLINE := 1

UV_RUN := uv run --frozen

.PHONY: sync check check-full lint format-check format typecheck test test-full doctor

sync:
	uv sync --frozen

# Quick check: everything except tests that load the translation model.
check: lint format-check typecheck test

# Full check, required at every phase gate and before every merge: includes the model tests
# and fails (never skips) when the local model files or fixture clips are missing.
check-full: lint format-check typecheck test-full

lint:
	$(UV_RUN) ruff check .

format-check:
	$(UV_RUN) ruff format --check .

format:
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

typecheck:
	$(UV_RUN) mypy

test:
	$(UV_RUN) pytest -m "not model"

test-full:
	S2E_REQUIRE_ASSETS=1 $(UV_RUN) pytest

doctor:
	$(UV_RUN) s2e doctor
