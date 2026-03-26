"""
Configurable retry policy with exponential backoff and jitter.

Replaces three independent retry implementations across the codebase
with a single reusable utility.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional, TypeVar

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


class RetryPolicy:
    """Execute a callable with configurable retry, backoff, and jitter.

    Args:
        max_retries: Maximum number of attempts (≥1).
        base_delay: Initial delay in seconds before first retry.
        max_delay: Cap on the computed delay.
        jitter: Maximum random jitter added to each delay.
        retry_on: Predicate that receives the caught exception and returns
            True if the operation should be retried.
        stop_event: Optional threading.Event checked between retries;
            if set, remaining retries are abandoned.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 8.0,
        jitter: float = 0.25,
        retry_on: Optional[Callable[[Exception], bool]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.max_retries = max(1, max_retries)
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on = retry_on or (lambda _: False)
        self.stop_event = stop_event

    def execute(self, fn: Callable[[], T], label: str = "operation") -> T:
        """Run *fn* up to ``max_retries`` times.

        Returns:
            The return value of *fn* on success.

        Raises:
            The last exception raised by *fn* if all retries are exhausted.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                can_retry = (
                    self.retry_on(exc)
                    and attempt < self.max_retries - 1
                    and (self.stop_event is None or not self.stop_event.is_set())
                )
                if can_retry:
                    delay = min(self.max_delay, self.base_delay * (2 ** attempt))
                    delay += random.uniform(0.0, self.jitter)
                    logger.warning(
                        "%s: retrying in %.2fs (attempt %d/%d): %s",
                        label,
                        delay,
                        attempt + 1,
                        self.max_retries,
                        exc,
                    )
                    if self.stop_event:
                        self.stop_event.wait(delay)
                    else:
                        time.sleep(delay)
                    continue
                raise
        # Should never reach here, but satisfy the type checker.
        raise last_exc  # type: ignore[misc]
