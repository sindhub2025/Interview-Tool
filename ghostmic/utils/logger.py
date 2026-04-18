"""
Centralised logging configuration for GhostMic.

Creates a rotating file handler (max 5 MB, 3 backups) and an optional
console handler for debug mode.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Iterator


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _candidate_log_dirs() -> Iterator[str]:
    override = os.environ.get("GHOSTMIC_LOG_DIR", "").strip()
    if override:
        yield os.path.abspath(override)

    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        yield os.path.join(exe_dir, "startup-logs")
    else:
        yield os.path.join(_project_root(), "startup-logs")

    yield os.path.join(os.path.expanduser("~"), ".ghostmic")


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe_path = os.path.join(path, ".ghostmic_write_probe")
        with open(probe_path, "a", encoding="utf-8"):
            pass
        try:
            os.remove(probe_path)
        except OSError:
            pass
        return True
    except OSError:
        return False


def _resolve_log_dir() -> str:
    for candidate in _candidate_log_dirs():
        if _is_writable_dir(candidate):
            return candidate
    return os.path.join(os.path.expanduser("~"), ".ghostmic")


_LOG_DIR = _resolve_log_dir()
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


def get_log_dir() -> str:
    """Return the directory where GhostMic diagnostic logs are stored."""
    return _LOG_DIR
