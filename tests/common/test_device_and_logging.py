from __future__ import annotations

import io
import json
import logging

import pytest

from signtoenglish.common import device
from signtoenglish.common.logging import ROOT_LOGGER, get_logger, setup_logging


def test_auto_selects_mps_on_this_machine_and_a_tensor_lands_there() -> None:
    import torch

    name = device.select_device("auto")
    assert name == "mps", "this project targets an Apple Silicon Mac; MPS should be available"
    assert torch.ones(2, dtype=torch.float32, device=name).device.type == "mps"


def test_cpu_is_always_selectable() -> None:
    assert device.select_device("cpu") == "cpu"


def test_auto_falls_back_to_cpu_when_mps_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "mps_available", lambda: False)
    assert device.select_device("auto") == "cpu"


def test_explicit_mps_does_not_fall_back_silently(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "mps_available", lambda: False)
    with pytest.raises(RuntimeError, match="mps"):
        device.select_device("mps")


@pytest.mark.parametrize("name", ["cuda", "gpu", ""])
def test_there_is_no_cuda_path(name: str) -> None:
    with pytest.raises(ValueError):
        device.select_device(name)  # type: ignore[arg-type]


def test_text_logging_and_logger_names() -> None:
    stream = io.StringIO()
    setup_logging("INFO", stream=stream)
    log = get_logger("preprocess.extract")
    assert log.name == f"{ROOT_LOGGER}.preprocess.extract"
    assert get_logger("signtoenglish.serve.app").name == "signtoenglish.serve.app"

    log.debug("hidden")
    log.info("kept %d", 3)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("INFO    signtoenglish.preprocess.extract: kept 3")


def test_json_logging_and_reconfiguration_replaces_the_handler() -> None:
    first, second = io.StringIO(), io.StringIO()
    setup_logging("INFO", stream=first)
    logger = setup_logging("DEBUG", json_lines=True, stream=second)
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG

    try:
        raise KeyError("boom")
    except KeyError:
        get_logger("x").exception("failed é")

    assert first.getvalue() == ""
    record = json.loads(second.getvalue())
    assert record["level"] == "ERROR"
    assert record["logger"] == "signtoenglish.x"
    assert record["message"] == "failed é"
    assert "KeyError" in record["exception"]
