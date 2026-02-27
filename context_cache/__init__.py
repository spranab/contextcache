"""ContextCache: KV cache routing for tool selection — 290x faster TTFT.

Multi-tenant tool routing with cached KV states. Register tool sets,
route queries to the right tool in ~150ms instead of ~58s.

SDK usage:
    from contextcache import ContextCacheClient

    client = ContextCacheClient("http://localhost:8421", api_key="your-key")
    client.register_tools("merchant", tools=[...])
    result = client.route("merchant", "What's my GMV?")
"""

from context_cache.cache_config import ContextCacheConfig
from context_cache.client import ContextCacheClient
from context_cache.kv_store import CachedContextKV, ContextKVStore
from context_cache.llm_adapter import (
    ClaudeAdapter,
    LLMAdapter,
    LLMResponse,
    OpenAIAdapter,
    get_llm_adapter,
)
from context_cache.llm_config import LLMConfig, LLMConfigStore
from context_cache.rope_utils import apply_rope, build_rope_cache, reverse_rope

__all__ = [
    "ContextCacheClient",
    "ContextCacheConfig",
    "ContextKVStore",
    "CachedContextKV",
    "LLMAdapter",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "LLMResponse",
    "get_llm_adapter",
    "LLMConfig",
    "LLMConfigStore",
    "build_rope_cache",
    "apply_rope",
    "reverse_rope",
]
