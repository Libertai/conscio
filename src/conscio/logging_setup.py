"""Process-wide logging configuration for the service path.

Stdout/stderr-first (12-factor): one stderr handler in text or JSON-lines
format, plus an optional rotating file. Uvicorn's loggers are forced to
propagate into the root handler so journald/docker see one consistent stream.
The interactive CLI (run/ask) deliberately does not call this — rich console
output is its interface.
"""

from __future__ import annotations

import json
import logging
import logging.config
import time
from pathlib import Path
from typing import Any

from conscio.config import ServiceConfig

_EXTRA_KEYS = ("event", "client", "tool", "episode_id")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in _EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(config: ServiceConfig) -> None:
    formatter = (
        {"()": "conscio.logging_setup.JsonFormatter"}
        if config.log_format == "json"
        else {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}
    )
    handlers: dict[str, dict[str, Any]] = {
        "stderr": {"class": "logging.StreamHandler", "stream": "ext://sys.stderr", "formatter": "main"}
    }
    root_handlers = ["stderr"]
    if config.log_file:
        Path(config.log_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(Path(config.log_file).expanduser()),
            "maxBytes": 10_485_760,
            "backupCount": 5,
            "formatter": "main",
        }
        root_handlers.append("file")
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"main": formatter},
            "handlers": handlers,
            "root": {"level": config.log_level.upper(), "handlers": root_handlers},
            "loggers": {
                # Single sink: uvicorn logs flow through the root handler.
                "uvicorn": {"handlers": [], "propagate": True},
                "uvicorn.error": {"handlers": [], "propagate": True},
                "uvicorn.access": {"handlers": [], "propagate": True},
            },
        }
    )
