"""ContextCache: KV cache routing for tool selection — 290x faster TTFT.

Multi-tenant tool routing with cached KV states. Register tool sets,
route queries to the right tool in ~150ms instead of ~58s.

SDK usage:
    from context_cache import ContextCacheClient

    client = ContextCacheClient("http://localhost:8421", api_key="your-key")
    client.register_tools("merchant", tools=[...])
    result = client.route("merchant", "What's my GMV?")
"""

# Submodule imports are deferred (PEP 562) so that `from context_cache import
# ContextCacheClient` — the documented SDK-only usage, which needs nothing
# beyond `requests` — doesn't force installation of the server-side deps
# (torch, numpy, pyyaml, llama-cpp) that other names here pull in.
_LAZY_ATTRS = {
    "ContextCacheConfig": "context_cache.cache_config",
    "ContextCacheClient": "context_cache.client",
    "CachedContextKV": "context_cache.kv_store",
    "ContextKVStore": "context_cache.kv_store",
    "ClaudeAdapter": "context_cache.llm_adapter",
    "LLMAdapter": "context_cache.llm_adapter",
    "LLMResponse": "context_cache.llm_adapter",
    "OpenAIAdapter": "context_cache.llm_adapter",
    "get_llm_adapter": "context_cache.llm_adapter",
    "LLMConfig": "context_cache.llm_config",
    "LLMConfigStore": "context_cache.llm_config",
    "CallableExecutor": "context_cache.orchestrator",
    "MockExecutor": "context_cache.orchestrator",
    "Orchestrator": "context_cache.orchestrator",
    "OrchestratorConfig": "context_cache.orchestrator",
    "OrchestratorResult": "context_cache.orchestrator",
    "ToolExecutor": "context_cache.orchestrator",
    "apply_rope": "context_cache.rope_utils",
    "build_rope_cache": "context_cache.rope_utils",
    "reverse_rope": "context_cache.rope_utils",
    "RouteResult": "context_cache.tool_router",
    "ToolRouter": "context_cache.tool_router",
}


def __getattr__(name):
    import importlib

    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))


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
    # Orchestrator
    "ToolRouter",
    "RouteResult",
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorResult",
    "ToolExecutor",
    "MockExecutor",
    "CallableExecutor",
]
