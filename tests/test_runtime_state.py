"""Unit tests for ghostmic.services.runtime_state — RuntimeStateCache."""

import json
import os
import time

import pytest

from ghostmic.services.runtime_state import RuntimeStateCache


@pytest.fixture
def cache_path(tmp_path):
    return str(tmp_path / ".runtime_state.json")


class TestRuntimeStateCache:
    def test_get_set(self, cache_path):
        cache = RuntimeStateCache(cache_path)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_missing_key_returns_default(self, cache_path):
        cache = RuntimeStateCache(cache_path)
        assert cache.get("missing") is None
        assert cache.get("missing", "fallback") == "fallback"

    def test_ttl_expiry(self, cache_path):
        cache = RuntimeStateCache(cache_path, default_ttl=0.1)
        cache.set("short_lived", "data")
        assert cache.get("short_lived") == "data"
        time.sleep(0.15)
        assert cache.get("short_lived") is None

    def test_custom_ttl_per_key(self, cache_path):
        cache = RuntimeStateCache(cache_path, default_ttl=3600)
        cache.set("expires_fast", "data", ttl=0.1)
        cache.set("expires_slow", "data", ttl=3600)
        time.sleep(0.15)
        assert cache.get("expires_fast") is None
        assert cache.get("expires_slow") == "data"

    def test_persistence(self, cache_path):
        cache1 = RuntimeStateCache(cache_path)
        cache1.set("persistent", 42)

        cache2 = RuntimeStateCache(cache_path)
        assert cache2.get("persistent") == 42

    def test_delete(self, cache_path):
        cache = RuntimeStateCache(cache_path)
        cache.set("key", "value")
        cache.delete("key")
        assert cache.get("key") is None

    def test_clear(self, cache_path):
        cache = RuntimeStateCache(cache_path)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_has(self, cache_path):
        cache = RuntimeStateCache(cache_path)
        assert cache.has("key") is False
        cache.set("key", "value")
        assert cache.has("key") is True

    def test_purge_expired(self, cache_path):
        cache = RuntimeStateCache(cache_path, default_ttl=0.1)
        cache.set("a", 1)
        cache.set("b", 2)
        time.sleep(0.15)
        purged = cache.purge_expired()
        assert purged == 2

    def test_corrupt_file_handled(self, cache_path):
        with open(cache_path, "w") as f:
            f.write("{bad json")
        cache = RuntimeStateCache(cache_path)
        assert cache.get("key") is None  # should not crash

    def test_complex_values(self, cache_path):
        cache = RuntimeStateCache(cache_path)
        cache.set("nested", {"list": [1, 2, 3], "dict": {"a": True}})
        result = cache.get("nested")
        assert result == {"list": [1, 2, 3], "dict": {"a": True}}
