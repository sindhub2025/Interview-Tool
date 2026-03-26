"""
Transient runtime state cache.

Stores data that should NOT be persisted in config.json — e.g. the
"known_local_torch_broken" flag, session counters, and health markers.

Uses a separate file with automatic TTL expiry for entries so stale
flags don't survive reinstalls.
"""

from __future__ import annotations

import json
import os
import time
import threading
from typing import Any, Dict, Optional

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)


class RuntimeStateCache:
    """Key/value cache with per-entry TTL, backed by a JSON file.

    Args:
        path: Path to the backing JSON file.
        default_ttl: Default time-to-live in seconds for entries (24 hours).
    """

    def __init__(self, path: str, default_ttl: float = 86400.0) -> None:
        self._path = path
        self._default_ttl = default_ttl
        self._state: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value. Returns *default* if key is missing or expired."""
        with self._lock:
            entry = self._state.get(key)
            if entry is None:
                return default
            if time.time() > entry.get("expires_at", 0):
                del self._state[key]
                self._save_unlocked()
                return default
            return entry.get("value", default)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value with optional TTL override."""
        with self._lock:
            self._state[key] = {
                "value": value,
                "expires_at": time.time() + (ttl if ttl is not None else self._default_ttl),
            }
            self._save_unlocked()

    def delete(self, key: str) -> None:
        """Remove a key."""
        with self._lock:
            self._state.pop(key, None)
            self._save_unlocked()

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._state.clear()
            self._save_unlocked()

    def has(self, key: str) -> bool:
        """Return True if key exists and is not expired."""
        return self.get(key, _SENTINEL) is not _SENTINEL

    def purge_expired(self) -> int:
        """Remove all expired entries. Returns the number of entries purged."""
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._state.items() if now > v.get("expires_at", 0)]
            for k in expired:
                del self._state[k]
            if expired:
                self._save_unlocked()
        return len(expired)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._state = data
                # Purge expired entries on load
                self.purge_expired()
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("RuntimeStateCache: could not load %s: %s", self._path, exc)
            self._state = {}

    def _save_unlocked(self) -> None:
        """Save state to disk. Caller must hold self._lock."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2)
        except OSError as exc:
            logger.warning("RuntimeStateCache: could not save %s: %s", self._path, exc)


_SENTINEL = object()
