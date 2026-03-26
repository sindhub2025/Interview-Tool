"""
Thread lifecycle coordinator for GhostMic.

Manages ordered startup and graceful shutdown of all background threads
with configurable timeouts, health checks, and orphan detection.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional, Protocol

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)


class Stoppable(Protocol):
    """Protocol for threads managed by ThreadCoordinator."""

    def stop(self) -> None: ...
    def isRunning(self) -> bool: ...
    def wait(self, msecs: int = ...) -> bool: ...


class ThreadCoordinator:
    """Manages ordered startup and graceful shutdown of background threads.

    Threads are registered with a name and started in registration order.
    Shutdown proceeds in reverse order with a configurable per-thread timeout.
    """

    def __init__(self) -> None:
        self._threads: Dict[str, Stoppable] = {}
        self._start_order: List[str] = []
        self._lock = threading.Lock()

    def register(self, name: str, thread: Stoppable) -> None:
        """Register a thread. Registration order determines startup order."""
        with self._lock:
            self._threads[name] = thread
            if name not in self._start_order:
                self._start_order.append(name)

    def start(self, name: str) -> bool:
        """Start a single registered thread by name.

        Returns True if the thread was started, False if already running
        or not registered.
        """
        with self._lock:
            t = self._threads.get(name)
        if t is None:
            logger.warning("ThreadCoordinator: unknown thread %r", name)
            return False
        if t.isRunning():
            return False
        if hasattr(t, "start"):
            t.start()  # type: ignore[attr-defined]
            logger.info("ThreadCoordinator: started %s", name)
            return True
        return False

    def start_all(self) -> int:
        """Start all registered threads in registration order.

        Returns the number of threads that were actually started.
        """
        started = 0
        with self._lock:
            order = list(self._start_order)
        for name in order:
            with self._lock:
                t = self._threads.get(name)
            if t and not t.isRunning() and hasattr(t, "start"):
                t.start()  # type: ignore[attr-defined]
                logger.info("ThreadCoordinator: started %s", name)
                started += 1
        return started

    def shutdown(self, timeout_ms: int = 5000) -> List[str]:
        """Graceful shutdown in reverse-start order.

        Args:
            timeout_ms: Maximum milliseconds to wait for each thread.

        Returns:
            Names of threads that did not stop within the timeout.
        """
        orphans: List[str] = []
        with self._lock:
            order = list(reversed(self._start_order))
        for name in order:
            with self._lock:
                t = self._threads.get(name)
            if t is None:
                continue
            if not t.isRunning():
                logger.debug("ThreadCoordinator: %s already stopped", name)
                continue
            t.stop()
            if hasattr(t, "wait") and not t.wait(timeout_ms):
                logger.error(
                    "ThreadCoordinator: %s did not stop within %dms",
                    name,
                    timeout_ms,
                )
                orphans.append(name)
            else:
                logger.info("ThreadCoordinator: %s stopped cleanly", name)
        return orphans

    def stop_one(self, name: str, timeout_ms: int = 3000) -> bool:
        """Stop a single thread by name.

        Returns True if the thread stopped within the timeout.
        """
        with self._lock:
            t = self._threads.get(name)
        if t is None or not t.isRunning():
            return True
        t.stop()
        if hasattr(t, "wait"):
            return bool(t.wait(timeout_ms))
        return True

    def is_alive(self, name: str) -> bool:
        """Return True if the named thread is currently running."""
        with self._lock:
            t = self._threads.get(name)
        return bool(t and t.isRunning())

    def health_check(self) -> Dict[str, bool]:
        """Return a dict mapping thread names to running status."""
        with self._lock:
            return {name: t.isRunning() for name, t in self._threads.items()}

    @property
    def registered_names(self) -> List[str]:
        """Return names of all registered threads in start order."""
        with self._lock:
            return list(self._start_order)
