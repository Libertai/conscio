"""JsonFormatter + setup_logging configuration."""

from __future__ import annotations

import json
import logging
import logging.handlers

from conscio.config import ServiceConfig
from conscio.logging_setup import JsonFormatter, setup_logging


def _record(msg: str, exc: bool = False) -> logging.LogRecord:
    exc_info = None
    if exc:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
    return logging.LogRecord("conscio.test", logging.WARNING, __file__, 1, msg, (), exc_info)


def test_json_formatter_shape() -> None:
    payload = json.loads(JsonFormatter().format(_record("hello")))
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "conscio.test"
    assert payload["msg"] == "hello"
    assert payload["ts"].endswith("Z")


def test_json_formatter_exception() -> None:
    payload = json.loads(JsonFormatter().format(_record("bad", exc=True)))
    assert "ValueError: boom" in payload["exc"]


def test_setup_logging_level_and_format(tmp_path) -> None:
    cfg = ServiceConfig(log_level="DEBUG", log_format="json", log_file=str(tmp_path / "logs" / "svc.log"))
    setup_logging(cfg)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    formatters = [type(h.formatter).__name__ for h in root.handlers]
    assert "JsonFormatter" in formatters
    file_handlers = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert file_handlers and file_handlers[0].maxBytes == 10_485_760 and file_handlers[0].backupCount == 5
    assert logging.getLogger("uvicorn.access").propagate is True
