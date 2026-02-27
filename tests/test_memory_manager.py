"""Tests for MemoryManager — GPU cache tracking and LRU eviction."""

import time

import pytest

from context_cache.memory_manager import MemoryManager


class TestMemoryManager:
    def test_register_and_total(self):
        mm = MemoryManager(max_gpu_cache_mb=1000)
        mm.register("key1", size_mb=200, tool_names=["tool_a"])
        mm.register("key2", size_mb=300, tool_names=["tool_b"])
        assert mm.total_mb == 500

    def test_over_budget(self):
        mm = MemoryManager(max_gpu_cache_mb=100)
        mm.register("key1", size_mb=60)
        assert not mm.over_budget
        mm.register("key2", size_mb=60)
        assert mm.over_budget

    def test_remove(self):
        mm = MemoryManager(max_gpu_cache_mb=1000)
        mm.register("key1", size_mb=200)
        mm.remove("key1")
        assert mm.total_mb == 0

    def test_remove_nonexistent(self):
        mm = MemoryManager(max_gpu_cache_mb=1000)
        mm.remove("nonexistent")  # Should not raise

    def test_touch_updates_access_time(self):
        mm = MemoryManager(max_gpu_cache_mb=1000)
        mm.register("key1", size_mb=100)
        old_time = mm._entries["key1"].last_accessed_at
        time.sleep(0.01)
        mm.touch("key1")
        new_time = mm._entries["key1"].last_accessed_at
        assert new_time > old_time

    def test_eviction_candidates_lru_order(self):
        mm = MemoryManager(max_gpu_cache_mb=250)
        mm.register("key1", size_mb=100)
        time.sleep(0.01)
        mm.register("key2", size_mb=100)
        time.sleep(0.01)
        mm.register("key3", size_mb=100)
        # Total: 300 > 250, so need to evict
        candidates = mm.eviction_candidates()
        # key1 was registered first (oldest), so it's the eviction candidate
        assert "key1" in candidates

    def test_eviction_candidates_respects_touch(self):
        mm = MemoryManager(max_gpu_cache_mb=250)
        mm.register("key1", size_mb=100)
        time.sleep(0.01)
        mm.register("key2", size_mb=100)
        time.sleep(0.01)
        mm.register("key3", size_mb=100)
        # Touch key1 to make it most recent
        mm.touch("key1")
        candidates = mm.eviction_candidates()
        # key2 should now be the candidate (oldest untouched)
        assert candidates[0] == "key2"

    def test_eviction_candidates_empty_when_under_budget(self):
        mm = MemoryManager(max_gpu_cache_mb=1000)
        mm.register("key1", size_mb=100)
        assert mm.eviction_candidates() == []

    def test_get_status(self):
        mm = MemoryManager(max_gpu_cache_mb=1000)
        mm.register("key1", size_mb=200, tool_names=["a", "b"])
        status = mm.get_status()
        assert status["total_cache_mb"] == 200
        assert status["max_cache_mb"] == 1000
        assert status["num_entries"] == 1
        assert status["utilization_pct"] == 20.0

    def test_register_overwrites(self):
        mm = MemoryManager(max_gpu_cache_mb=1000)
        mm.register("key1", size_mb=200)
        mm.register("key1", size_mb=300)
        assert mm.total_mb == 300
