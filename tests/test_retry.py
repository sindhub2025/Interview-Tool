"""Unit tests for ghostmic.utils.retry — the RetryPolicy."""

import threading
import time

import pytest

from ghostmic.utils.retry import RetryPolicy


class TestRetryPolicy:
    """Tests for RetryPolicy.execute()."""

    def test_succeeds_first_attempt(self) -> None:
        policy = RetryPolicy(max_retries=3)
        result = policy.execute(lambda: 42, label="test")
        assert result == 42

    def test_retries_on_matching_exception(self) -> None:
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("rate limit")
            return "success"

        policy = RetryPolicy(
            max_retries=5,
            base_delay=0.01,
            retry_on=lambda exc: "rate limit" in str(exc),
        )
        result = policy.execute(flaky, label="flaky_fn")
        assert result == "success"
        assert call_count == 3

    def test_raises_on_non_retryable_exception(self) -> None:
        policy = RetryPolicy(
            max_retries=3,
            retry_on=lambda exc: False,
        )
        with pytest.raises(ValueError, match="fatal"):
            policy.execute(lambda: (_ for _ in ()).throw(ValueError("fatal")), label="test")

    def test_raises_after_max_retries_exhausted(self) -> None:
        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("always fails")

        policy = RetryPolicy(
            max_retries=3,
            base_delay=0.01,
            retry_on=lambda exc: True,
        )
        with pytest.raises(RuntimeError, match="always fails"):
            policy.execute(always_fail, label="always_fail")
        assert call_count == 3

    def test_stop_event_cancels_retries(self) -> None:
        stop = threading.Event()
        stop.set()  # Already set — should not retry

        call_count = 0

        def failing():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        policy = RetryPolicy(
            max_retries=5,
            base_delay=0.01,
            retry_on=lambda exc: True,
            stop_event=stop,
        )
        with pytest.raises(RuntimeError):
            policy.execute(failing, label="stopped")
        # Should only execute once because stop_event is set
        assert call_count == 1

    def test_exponential_backoff(self) -> None:
        """Verify that delay increases exponentially."""
        delays = []

        orig_wait = threading.Event.wait

        class TimingEvent(threading.Event):
            def wait(self_, timeout=None):
                if timeout is not None:
                    delays.append(timeout)
                return orig_wait(self_, timeout)

        stop = TimingEvent()
        call_count = 0

        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("fail")
            return "ok"

        policy = RetryPolicy(
            max_retries=5,
            base_delay=0.1,
            max_delay=10.0,
            jitter=0.0,
            retry_on=lambda exc: True,
            stop_event=stop,
        )
        result = policy.execute(fail_twice, label="backoff_test")
        assert result == "ok"
        assert len(delays) == 2
        # First delay: 0.1 * 2^0 = 0.1, second: 0.1 * 2^1 = 0.2
        assert abs(delays[0] - 0.1) < 0.05
        assert abs(delays[1] - 0.2) < 0.05

    def test_max_retries_at_least_one(self) -> None:
        policy = RetryPolicy(max_retries=0)
        assert policy.max_retries == 1
