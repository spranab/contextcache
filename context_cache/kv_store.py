"""Persistent hash-addressed KV cache store.

Stores NoPE (No Positional Encoding) KV states on disk with content-hash keys.
When content changes, the hash changes -> automatic cache invalidation.

Storage format:
    cache_dir/
        index.json          # {hash: {name, content_type, num_tokens, file}}
        kv_{hash[:16]}.pt   # {keys: [L tensors], values: [L tensors], num_tokens: int}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class CachedContextKV:
    """A single cached context block's NoPE KV states."""

    name: str  # tool name, file path, doc title, etc.
    content_hash: str  # SHA256 of content text
    content_type: str  # "tool_schema" | "code_file" | "document" | "system_prompt"
    num_tokens: int  # number of real tokens (excluding dummy BOS)
    keys: list[Tensor] = field(repr=False)  # per-layer, each (1, num_kv_heads, num_tokens, head_dim)
    values: list[Tensor] = field(repr=False)  # per-layer, each (1, num_kv_heads, num_tokens, head_dim)


class ContextKVStore:
    """Persistent hash-addressed NoPE KV cache store.

    Keys and values are stored per-layer as separate tensors on disk.
    Content is addressed by SHA256 hash of the source text, so any
    change in content automatically invalidates the cache.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        num_layers: int = 36,
        device: str = "cpu",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_layers = num_layers
        self.default_device = device
        self._index: dict[str, dict] = {}
        self._gpu_cache: dict[str, CachedContextKV] = {}  # hash -> loaded KV on GPU
        self._load_index()

    def _index_path(self) -> Path:
        return self.cache_dir / "index.json"

    def _load_index(self):
        idx_path = self._index_path()
        if idx_path.exists():
            with open(idx_path, "r") as f:
                self._index = json.load(f)

    def _save_index(self):
        with open(self._index_path(), "w") as f:
            json.dump(self._index, f, indent=2)

    @staticmethod
    def hash_content(text: str) -> str:
        """SHA256 of content text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def has(self, text: str) -> bool:
        """Check if content's KV is cached (by content hash)."""
        h = self.hash_content(text)
        return h in self._index

    def get(self, text: str, device: str | None = None) -> CachedContextKV | None:
        """Retrieve cached NoPE KV. Returns None on miss."""
        h = self.hash_content(text)

        # Check GPU cache first
        if h in self._gpu_cache:
            cached = self._gpu_cache[h]
            target_device = device or self.default_device
            if str(cached.keys[0].device) != target_device:
                # Move to requested device
                return CachedContextKV(
                    name=cached.name,
                    content_hash=cached.content_hash,
                    content_type=cached.content_type,
                    num_tokens=cached.num_tokens,
                    keys=[k.to(target_device) for k in cached.keys],
                    values=[v.to(target_device) for v in cached.values],
                )
            return cached

        # Check disk index
        if h not in self._index:
            return None

        entry = self._index[h]
        kv_path = self.cache_dir / entry["file"]
        if not kv_path.exists():
            # Stale index entry
            del self._index[h]
            self._save_index()
            return None

        target_device = device or self.default_device
        data = torch.load(kv_path, map_location=target_device, weights_only=True)

        return CachedContextKV(
            name=entry["name"],
            content_hash=h,
            content_type=entry["content_type"],
            num_tokens=data["num_tokens"],
            keys=data["keys"],
            values=data["values"],
        )

    def put(
        self,
        name: str,
        text: str,
        content_type: str,
        keys: list[Tensor],
        values: list[Tensor],
        num_tokens: int,
    ):
        """Store NoPE KV to disk."""
        h = self.hash_content(text)
        filename = f"kv_{h[:16]}.pt"
        kv_path = self.cache_dir / filename

        # Save tensors to disk (move to CPU for storage)
        torch.save(
            {
                "keys": [k.cpu() for k in keys],
                "values": [v.cpu() for v in values],
                "num_tokens": num_tokens,
            },
            kv_path,
        )

        # Update index
        self._index[h] = {
            "name": name,
            "content_type": content_type,
            "num_tokens": num_tokens,
            "file": filename,
        }
        self._save_index()

    def invalidate(self, text: str):
        """Remove cached entry explicitly."""
        h = self.hash_content(text)
        if h in self._index:
            entry = self._index.pop(h)
            kv_path = self.cache_dir / entry["file"]
            if kv_path.exists():
                kv_path.unlink()
            self._save_index()
        self._gpu_cache.pop(h, None)

    def list_entries(self) -> list[dict]:
        """List all cached entries."""
        entries = []
        for h, entry in self._index.items():
            entries.append(
                {
                    "content_hash": h,
                    "name": entry["name"],
                    "content_type": entry["content_type"],
                    "num_tokens": entry["num_tokens"],
                }
            )
        return entries

    def preload_to_gpu(self, names: list[str] | None = None, device: str = "cuda"):
        """Preload entries' KV to GPU memory for fast linking.

        Args:
            names: List of names to preload. If None, preload all.
            device: GPU device string.
        """
        for h, entry in self._index.items():
            if names is not None and entry["name"] not in names:
                continue
            if h in self._gpu_cache:
                continue
            kv_path = self.cache_dir / entry["file"]
            if not kv_path.exists():
                continue
            data = torch.load(kv_path, map_location=device, weights_only=True)
            self._gpu_cache[h] = CachedContextKV(
                name=entry["name"],
                content_hash=h,
                content_type=entry["content_type"],
                num_tokens=data["num_tokens"],
                keys=data["keys"],
                values=data["values"],
            )

    def gpu_cache_size_mb(self) -> float:
        """Estimate GPU memory used by cached entries."""
        total_bytes = 0
        for cached in self._gpu_cache.values():
            for k in cached.keys:
                total_bytes += k.nelement() * k.element_size()
            for v in cached.values:
                total_bytes += v.nelement() * v.element_size()
        return total_bytes / (1024 * 1024)

    def clear_gpu_cache(self):
        """Release all GPU-cached KV states."""
        self._gpu_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, text: str) -> bool:
        return self.has(text)
