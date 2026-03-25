"""
Centralised logging configuration for GhostMic.

Creates a rotating file handler (max 5 MB, 3 backups) and an optional
console handler for debug mode.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

_LOG_DIR = os.path.join(os.path.expanduser("~"), ".ghostmic")
_LOG_FILE = os.path.join(_LOG_DIR, "ghostmic.log")
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_root_configured = False


def configure_logging(debug: bool = False) -> None:
    """Set up root logger with file and optional console handlers.

    Should be called once at application startup.

    Args:
        debug: When True, also output DEBUG messages to the console.
    """
    global _root_configured  # noqa: PLW0603
    if _root_configured:
        return

    os.makedirs(_LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # Rotating file handler — always at DEBUG level
    try:
        fh = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
    except OSError as exc:
        # Can't create log file – warn to stderr and continue
        import sys
        print(f"[GhostMic] WARNING: cannot open log file: {exc}", file=sys.stderr)

    # Console handler — only in debug mode
    if debug:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  No-op if configure_logging has not been called.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


def get_log_file_path() -> str:
    """Return the absolute path of the GhostMic log file."""
    return _LOG_FILE
