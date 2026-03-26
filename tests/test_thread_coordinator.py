"""Unit tests for ghostmic.services.thread_coordinator — ThreadCoordinator."""

import threading
import time

import pytest

from ghostmic.services.thread_coordinator import ThreadCoordinator


class FakeThread:
    """Minimal Stoppable implementation for testing."""

    def __init__(self, name: str = "fake", hang: bool = False):
        self._name = name
        self._running = False
        self._hang = hang
        self._stop_called = False

    def start(self):
        self._running = True

    def stop(self):
        self._stop_called = True
        if not self._hang:
            self._running = False

    def isRunning(self) -> bool:
        return self._running

    def wait(self, msecs: int = 5000) -> bool:
        if self._hang:
            # Simulate a thread that doesn't stop
            time.sleep(msecs / 1000.0)
            return False
        return True


class TestThreadCoordinator:
    def test_register_and_start_all(self):
        coord = ThreadCoordinator()
        t1 = FakeThread("a")
        t2 = FakeThread("b")
        coord.register("a", t1)
        coord.register("b", t2)
        started = coord.start_all()
        assert started == 2
        assert t1.isRunning()
        assert t2.isRunning()

    def test_start_single(self):
        coord = ThreadCoordinator()
        t = FakeThread("solo")
        coord.register("solo", t)
        assert coord.start("solo") is True
        assert t.isRunning()

    def test_start_already_running(self):
        coord = ThreadCoordinator()
        t = FakeThread("running")
        t.start()
        coord.register("running", t)
        assert coord.start("running") is False  # already running

    def test_start_unknown(self):
        coord = ThreadCoordinator()
        assert coord.start("unknown") is False

    def test_shutdown_reverse_order(self):
        stop_order = []

        class OrderedThread(FakeThread):
            def stop(self):
                stop_order.append(self._name)
                super().stop()

        coord = ThreadCoordinator()
        t1 = OrderedThread("first")
        t2 = OrderedThread("second")
        t3 = OrderedThread("third")
        coord.register("first", t1)
        coord.register("second", t2)
        coord.register("third", t3)
        coord.start_all()

        orphans = coord.shutdown(timeout_ms=1000)
        assert orphans == []
        assert stop_order == ["third", "second", "first"]

    def test_shutdown_detects_orphans(self):
        coord = ThreadCoordinator()
        t = FakeThread("hanging", hang=True)
        coord.register("hanging", t)
        t.start()

        # Use very short timeout to avoid slow test
        orphans = coord.shutdown(timeout_ms=50)
        assert "hanging" in orphans

    def test_stop_one(self):
        coord = ThreadCoordinator()
        t = FakeThread("stoppable")
        coord.register("stoppable", t)
        t.start()
        assert coord.stop_one("stoppable", timeout_ms=1000) is True
        assert not t.isRunning()

    def test_is_alive(self):
        coord = ThreadCoordinator()
        t = FakeThread("test")
        coord.register("test", t)
        assert coord.is_alive("test") is False
        t.start()
        assert coord.is_alive("test") is True

    def test_health_check(self):
        coord = ThreadCoordinator()
        t1 = FakeThread("alive")
        t2 = FakeThread("dead")
        coord.register("alive", t1)
        coord.register("dead", t2)
        t1.start()

        health = coord.health_check()
        assert health == {"alive": True, "dead": False}

    def test_registered_names(self):
        coord = ThreadCoordinator()
        coord.register("a", FakeThread())
        coord.register("b", FakeThread())
        coord.register("c", FakeThread())
        assert coord.registered_names == ["a", "b", "c"]
