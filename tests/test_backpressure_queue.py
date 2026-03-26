"""Unit tests for ghostmic.utils.backpressure_queue — BackpressureQueue."""

import queue
import threading

import pytest

from ghostmic.utils.backpressure_queue import BackpressureQueue


class TestBackpressureQueue:
    def test_basic_put_get(self):
        bq = BackpressureQueue(maxsize=5)
        assert bq.put("a") is True
        assert bq.get(timeout=0.1) == "a"

    def test_drop_oldest_on_overflow(self):
        bq = BackpressureQueue(maxsize=3)
        bq.put("a")
        bq.put("b")
        bq.put("c")
        # Queue is now full; putting "d" should drop "a"
        assert bq.put("d") is False
        items = [bq.get(timeout=0.1) for _ in range(3)]
        assert items == ["b", "c", "d"]

    def test_drop_count(self):
        bq = BackpressureQueue(maxsize=2)
        bq.put("a")
        bq.put("b")
        bq.put("c")  # drop
        bq.put("d")  # drop
        assert bq.drop_count == 2

    def test_on_drop_callback(self):
        drops = []
        bq = BackpressureQueue(maxsize=1, on_drop=drops.append)
        bq.put("a")
        bq.put("b")  # triggers callback
        assert drops == [1]
        bq.put("c")  # triggers again
        assert drops == [1, 2]

    def test_get_timeout_raises_empty(self):
        bq = BackpressureQueue(maxsize=5)
        with pytest.raises(queue.Empty):
            bq.get(timeout=0.01)

    def test_qsize(self):
        bq = BackpressureQueue(maxsize=10)
        bq.put("a")
        bq.put("b")
        assert bq.qsize() == 2

    def test_empty(self):
        bq = BackpressureQueue(maxsize=5)
        assert bq.empty() is True
        bq.put("a")
        assert bq.empty() is False

    def test_clear(self):
        bq = BackpressureQueue(maxsize=10)
        bq.put("a")
        bq.put("b")
        bq.put("c")
        removed = bq.clear()
        assert removed == 3
        assert bq.empty()

    def test_reset_drop_count(self):
        bq = BackpressureQueue(maxsize=1)
        bq.put("a")
        bq.put("b")  # drop
        assert bq.drop_count == 1
        bq.reset_drop_count()
        assert bq.drop_count == 0

    def test_thread_safety(self):
        """Verify no crashes under concurrent access."""
        bq = BackpressureQueue(maxsize=10, name="stress")
        errors = []

        def producer():
            try:
                for i in range(100):
                    bq.put(i)
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                for _ in range(50):
                    try:
                        bq.get(timeout=0.01)
                    except queue.Empty:
                        pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=producer) for _ in range(3)]
        threads += [threading.Thread(target=consumer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(errors) == 0
