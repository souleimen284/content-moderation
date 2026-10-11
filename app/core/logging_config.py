"""
Application-wide logging configuration.

Every module should obtain a logger via ``get_logger(__name__)`` rather than
using ``print`` or ``logging.getLogger`` directly. This module configures:

  - Console logging (human-readable in development, JSON in production)
  - Rotating file logging under ``settings.log_dir``
  - Consistent formatting including timestamp, level, logger name, and
    request/correlation IDs when available.

Call ``configure_logging()`` once at process startup (done automatically by
``app.main`` and ``dashboard/app.py``); every subsequent ``get_logger`` call
just returns a standard ``logging.Logger`` bound to that configuration.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON, suited for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Idempotently configure root logging handlers based on ``settings``."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())
    root_logger.handlers.clear()

    if settings.log_json:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Quiet noisy third-party loggers unless we're debugging.
    if not settings.debug:
        for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
