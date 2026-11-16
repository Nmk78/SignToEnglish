# One entry point for every check. Uses the locked environment only; never the system Python.
# Model hubs are offline by default (CLAUDE.md, stop point 2).
export HF_HUB_OFFLINE := 1
export TRANSFORMERS_OFFLINE := 1

UV_RUN := uv run --frozen

.PHONY: sync check lint format-check format typecheck test doctor

sync:
	uv sync --frozen

check: lint format-check typecheck test

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
	$(UV_RUN) pytest

doctor:
	$(UV_RUN) s2e doctor
