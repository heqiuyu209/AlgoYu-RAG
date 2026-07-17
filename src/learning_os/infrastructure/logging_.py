"""
Unified logging for Learning OS.

Sets up the root logger so every module gets consistent formatting
and level control from a single place.  No other module should call
``logging.basicConfig`` — they only import the standard ``logging``
module and call ``logging.getLogger(__name__)``.

Usage::

    from learning_os.infrastructure.logging_ import setup_logging
    setup_logging(config_dict)
"""

from __future__ import annotations

import logging
import logging.config
import sys
from typing import Any

_TEXT_FORMAT = (
    "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s"
)
_TEXT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(config: dict[str, Any]) -> None:
    """Configure the Python logging subsystem from the project config.

    Args:
        config: The ``learning_os.logging`` section of ``config.yaml``,
            e.g. ``{"level": "INFO", "format": "text", ...}``.
    """
    level_name = config.get("level", "INFO").upper()
    file_enabled = config.get("file_enabled", False)
    file_path = config.get("file_path", "./logs/learning_os.log")

    handlers: dict[str, dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level_name,
            "formatter": "text",
            "stream": sys.stderr,
        }
    }

    formatters: dict[str, dict[str, Any]] = {
        "text": {
            "format": _TEXT_FORMAT,
            "datefmt": _TEXT_DATE_FORMAT,
        },
    }

    if file_enabled:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level_name,
            "formatter": "text",
            "filename": file_path,
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": handlers,
            "root": {
                "level": level_name,
                "handlers": list(handlers.keys()),
            },
        }
    )

    # Suppress overly chatty third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
