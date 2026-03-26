"""
Queue with drop-oldest policy and observable backpressure.

Used by TranscriptionThread and AIThread to replace ad-hoc queue
overflow handling with a consistent, signal-emitting implementation.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Optional, TypeVar

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


class BackpressureQueue:
    """Thread-safe queue with drop-oldest overflow and drop notifications.

    Args:
        maxsize: Maximum number of items in the queue.
        on_drop: Optional callback invoked with the cumulative drop count
            each time an item is dropped.  Called from the producer thread.
        name: Human-readable identifier for logging.
    """

    def __init__(
        self,
        maxsize: int,
        on_drop: Optional[Callable[[int], None]] = None,
        name: str = "queue",
    ) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._on_drop = on_drop
        self._drop_count = 0
        self._lock = threading.Lock()
        self._name = name

    def put(self, item: Any) -> bool:
        """Add an item; drop the oldest if full.

        Returns True if accepted without drop, False if an item was evicted.
        """
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            # Drop oldest
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return False
            self._queue.put_nowait(item)
            with self._lock:
                self._drop_count += 1
                count = self._drop_count
            logger.debug("%s: dropped oldest item (total drops: %d)", self._name, count)
            if self._on_drop:
                try:
                    self._on_drop(count)
                except Exception:  # pylint: disable=broad-except
                    pass
            return False

    def get(self, timeout: float = 0.2) -> Any:
        """Get an item from the queue with timeout.

        Raises:
            queue.Empty: If no item is available within the timeout.
        """
        return self._queue.get(timeout=timeout)

    def qsize(self) -> int:
        """Return the approximate size of the queue."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return self._queue.empty()

    def clear(self) -> int:
        """Remove all items from the queue. Returns number of items removed."""
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count

    @property
    def drop_count(self) -> int:
        """Cumulative number of items dropped since creation."""
        with self._lock:
            return self._drop_count

    def reset_drop_count(self) -> None:
        """Reset the drop counter to zero."""
        with self._lock:
            self._drop_count = 0
