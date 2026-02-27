"""Tests for production middleware — API key store, rate limiter, metrics."""

import json
import tempfile
import time

import pytest

from context_cache.middleware import APIKeyStore, MetricsCollector, SlidingWindowRateLimiter


# ---------------------------------------------------------------------------
# APIKeyStore
# ---------------------------------------------------------------------------


class TestAPIKeyStore:
    def test_add_and_validate(self):
        store = APIKeyStore()
        store.add_key("key-123", name="test", rate_limit=100)
        info = store.validate("key-123")
        assert info is not None
        assert info["name"] == "test"
        assert info["rate_limit"] == 100

    def test_validate_missing_key(self):
        store = APIKeyStore()
        assert store.validate("nonexistent") is None

    def test_remove_key(self):
        store = APIKeyStore()
        store.add_key("key-123", name="test")
        store.remove_key("key-123")
        assert store.validate("key-123") is None

    def test_remove_nonexistent_key(self):
        store = APIKeyStore()
        store.remove_key("nonexistent")  # Should not raise

    def test_num_keys(self):
        store = APIKeyStore()
        assert store.num_keys == 0
        store.add_key("k1")
        store.add_key("k2")
        assert store.num_keys == 2

    def test_from_json(self):
        data = {"keys": [
            {"key": "abc", "name": "alice", "rate_limit": 200},
            {"key": "def", "name": "bob"},
        ]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            store = APIKeyStore.from_json(f.name)

        assert store.num_keys == 2
        assert store.validate("abc")["rate_limit"] == 200
        assert store.validate("def")["name"] == "bob"

    def test_from_env_key(self):
        store = APIKeyStore.from_env_key("env-secret")
        assert store.num_keys == 1
        info = store.validate("env-secret")
        assert info["name"] == "env_key"
        assert info["rate_limit"] == 120

    def test_default_rate_limit(self):
        store = APIKeyStore()
        store.add_key("key-123")
        assert store.validate("key-123")["rate_limit"] == 60


# ---------------------------------------------------------------------------
# SlidingWindowRateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = SlidingWindowRateLimiter(default_rpm=10)
        allowed, remaining = limiter.check("user1")
        assert allowed is True
        assert remaining == 9

    def test_blocks_at_limit(self):
        limiter = SlidingWindowRateLimiter(default_rpm=3)
        for _ in range(3):
            limiter.check("user1")
        allowed, remaining = limiter.check("user1")
        assert allowed is False
        assert remaining == 0

    def test_custom_limit_per_key(self):
        limiter = SlidingWindowRateLimiter(default_rpm=100)
        for _ in range(2):
            limiter.check("user1", limit=2)
        allowed, _ = limiter.check("user1", limit=2)
        assert allowed is False

    def test_different_keys_independent(self):
        limiter = SlidingWindowRateLimiter(default_rpm=2)
        limiter.check("user1")
        limiter.check("user1")
        # user1 is at limit
        allowed1, _ = limiter.check("user1")
        assert allowed1 is False
        # user2 is fresh
        allowed2, _ = limiter.check("user2")
        assert allowed2 is True

    def test_remaining_decrements(self):
        limiter = SlidingWindowRateLimiter(default_rpm=5)
        _, r1 = limiter.check("k")
        _, r2 = limiter.check("k")
        _, r3 = limiter.check("k")
        assert r1 == 4
        assert r2 == 3
        assert r3 == 2


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def test_record_request(self):
        mc = MetricsCollector()
        mc.record_request("k1", latency_ms=50.0, cache_hit=True)
        m = mc.get_metrics("k1")
        assert m["total_requests"] == 1
        assert m["cache_hits"] == 1
        assert m["total_latency_ms"] == 50.0

    def test_record_error(self):
        mc = MetricsCollector()
        mc.record_request("k1", latency_ms=10.0, error=True)
        m = mc.get_metrics("k1")
        assert m["errors"] == 1

    def test_avg_latency(self):
        mc = MetricsCollector()
        mc.record_request("k1", latency_ms=100.0)
        mc.record_request("k1", latency_ms=200.0)
        m = mc.get_metrics("k1")
        assert m["avg_latency_ms"] == 150.0

    def test_global_metrics(self):
        mc = MetricsCollector()
        mc.record_request("k1", latency_ms=100.0, cache_hit=True)
        mc.record_request("k2", latency_ms=200.0)
        m = mc.get_metrics()
        assert m["total_requests"] == 2
        assert m["cache_hits"] == 1
        assert m["num_keys"] == 2
        assert "per_key" in m

    def test_unknown_key_returns_empty(self):
        mc = MetricsCollector()
        assert mc.get_metrics("nonexistent") == {}

    def test_cache_miss_counted(self):
        mc = MetricsCollector()
        mc.record_request("k1", latency_ms=50.0, cache_hit=False)
        m = mc.get_metrics("k1")
        assert m["cache_misses"] == 1
        assert m["cache_hits"] == 0
