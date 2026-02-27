"""GPU memory tracking and LRU eviction for ContextCache group caches."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """Tracks metadata for one group cache entry."""
    group_key: str
    size_mb: float
    tool_names: list[str]
    created_at: float
    last_accessed_at: float


class MemoryManager:
    """Tracks GPU cache memory usage and evicts LRU entries when over budget.

    Works with ContextCacheModel._group_cache to enforce memory limits.
    """

    def __init__(self, max_gpu_cache_mb: float = 4000):
        self.max_gpu_cache_mb = max_gpu_cache_mb
        self._entries: dict[str, CacheEntry] = {}

    def register(self, group_key: str, size_mb: float, tool_names: list[str] | None = None):
        now = time.time()
        self._entries[group_key] = CacheEntry(
            group_key=group_key,
            size_mb=size_mb,
            tool_names=tool_names or [],
            created_at=now,
            last_accessed_at=now,
        )

    def touch(self, group_key: str):
        entry = self._entries.get(group_key)
        if entry:
            entry.last_accessed_at = time.time()

    def remove(self, group_key: str):
        self._entries.pop(group_key, None)

    @property
    def total_mb(self) -> float:
        return sum(e.size_mb for e in self._entries.values())

    @property
    def over_budget(self) -> bool:
        return self.total_mb > self.max_gpu_cache_mb

    def eviction_candidates(self) -> list[str]:
        """Return group keys sorted by LRU (oldest access first)."""
        sorted_entries = sorted(
            self._entries.values(), key=lambda e: e.last_accessed_at
        )
        candidates = []
        total = self.total_mb
        for entry in sorted_entries:
            if total <= self.max_gpu_cache_mb:
                break
            candidates.append(entry.group_key)
            total -= entry.size_mb
        return candidates

    def get_status(self) -> dict:
        return {
            "total_cache_mb": round(self.total_mb, 1),
            "max_cache_mb": self.max_gpu_cache_mb,
            "utilization_pct": round(self.total_mb / max(self.max_gpu_cache_mb, 1) * 100, 1),
            "num_entries": len(self._entries),
            "entries": {
                k[:16]: {
                    "size_mb": round(e.size_mb, 1),
                    "tool_count": len(e.tool_names),
                    "last_accessed": e.last_accessed_at,
                }
                for k, e in self._entries.items()
            },
        }
