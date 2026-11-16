"""Logging setup shared by the CLI, scripts, and the service."""

from __future__ import annotations

import json
import logging
import sys
from typing import IO

ROOT_LOGGER = "signtoenglish"
_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, and exception if any."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    level: int | str = "INFO", *, json_lines: bool = False, stream: IO[str] | None = None
) -> logging.Logger:
    """Configure the project logger. Calling it again replaces the previous handler."""
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter() if json_lines else logging.Formatter(_TEXT_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Logger under the project root, e.g. `get_logger(__name__)`."""
    if name == ROOT_LOGGER or name.startswith(ROOT_LOGGER + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")
