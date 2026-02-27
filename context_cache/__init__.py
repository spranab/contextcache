"""ContextCache: Persistent, composable KV cache with content-hash addressing.

Enables position-independent caching of any context block (tool schemas, code
files, documents) with deferred RoPE for lossless composition at inference time.
"""

from context_cache.cache_config import ContextCacheConfig
from context_cache.kv_store import CachedContextKV, ContextKVStore
from context_cache.rope_utils import apply_rope, build_rope_cache, reverse_rope

__all__ = [
    "ContextCacheConfig",
    "ContextKVStore",
    "CachedContextKV",
    "build_rope_cache",
    "apply_rope",
    "reverse_rope",
]
