#!/usr/bin/env python3
"""FastAPI server for ContextCache — multi-tenant tool routing with cached KV.

Supports two modes:
  Live mode:  Full model inference with real KV cache (requires GPU)
  Demo mode:  Pre-recorded responses with simulated timings (no GPU needed)

V1 Endpoints (single-tenant, backward compatible):
  POST /tools          Register tool schemas -> compile group KV cache
  POST /query          Run inference with cached tools
  POST /query/compare  A/B comparison: cached vs full prefill timings
  GET  /status         Current cache state, loaded tools, memory usage
  DELETE /tools        Clear cached tools

V2 Endpoints (multi-tenant):
  POST /v2/tools              Register a named tool set (tool_id)
  POST /route                 Tool selection only — returns {tool_name, arguments}
  POST /v2/query              Raw query against a named tool set
  POST /v2/pipeline           Full pipeline: route → execute → format for LLM
  GET  /v2/registry           List all registered tool sets and stats
  DELETE /v2/tools/{tool_id}  Remove a specific tool set

Admin Endpoints (require auth):
  GET  /admin/metrics         Request metrics per API key
  GET  /admin/memory          GPU cache memory status
  POST /admin/evict           Force evict LRU caches

Common Endpoints:
  GET  /health         Health check (works during model loading)
  GET  /sample-tools   Returns demo tool schemas
  GET  /profiles       List available team profiles
  GET  /mode           Current server mode (live or demo)

Usage:
  python scripts/serve/serve_context_cache.py                    # Live mode (no auth)
  python scripts/serve/serve_context_cache.py --demo             # Demo mode
  python scripts/serve/serve_context_cache.py --auth             # Enable API key auth
  python scripts/serve/serve_context_cache.py --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
SERVE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ToolsRequest(BaseModel):
    tools: list[dict] = Field(
        ...,
        description="List of tool schemas in OpenAI function-calling format.",
    )


class ToolsResponse(BaseModel):
    status: str
    num_tools: int
    cache_hash: str
    compile_ms: float
    cache_size_mb: float
    prefix_tokens: int
    from_disk: bool = False


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text")
    max_new_tokens: int = Field(256, ge=1, le=2048)


class QueryResponse(BaseModel):
    response: str
    cache_hit: bool
    timings: dict


class CompareResponse(BaseModel):
    response: str
    cached_timings: dict
    full_prefill_timings: dict
    speedup: float


# --- V2 multi-tenant models ---


class ToolsRequestV2(BaseModel):
    tool_id: str = Field(..., description="Unique namespace for this tool set")
    tools: list[dict] = Field(..., description="Tool schemas in OpenAI function-calling format")


class ToolsResponseV2(BaseModel):
    status: str
    tool_id: str
    num_tools: int
    cache_hash: str
    compile_ms: float
    cache_size_mb: float
    prefix_tokens: int
    from_disk: bool = False


class RouteRequest(BaseModel):
    tool_id: str = Field(..., description="Which tool set to route against")
    query: str = Field(..., description="User query text")
    max_new_tokens: int = Field(256, ge=1, le=2048)


class ToolSelection(BaseModel):
    tool_name: str
    arguments: dict


class RouteResponse(BaseModel):
    tool_id: str
    selections: list[ToolSelection]
    raw_response: str
    confidence: float
    timings: dict


class QueryRequestV2(BaseModel):
    tool_id: str = Field(..., description="Which tool set to query against")
    query: str = Field(..., description="User query text")
    max_new_tokens: int = Field(256, ge=1, le=2048)


class QueryResponseV2(BaseModel):
    response: str
    tool_id: str
    cache_hit: bool
    timings: dict


class PipelineRequest(BaseModel):
    tool_id: str = Field(..., description="Which tool set to route against")
    query: str = Field(..., description="User query text")
    llm_format: str = Field("claude", description="LLM provider/format: 'claude', 'openai', or 'raw'")
    tool_executor: str = Field("mock", description="'mock' for simulated results, or URL for real execution")
    max_new_tokens: int = Field(256, ge=1, le=2048)
    llm_api_key: str | None = Field(None, description="API key for external LLM. If set, calls the LLM and returns final answer.")
    llm_model: str | None = Field(None, description="LLM model to use (e.g. 'claude-sonnet-4-20250514', 'gpt-4o'). Uses provider default if not set.")
    llm_system_prompt: str | None = Field(None, description="System prompt for the enterprise LLM")
    llm_base_url: str | None = Field(None, description="Custom LLM endpoint URL (enterprise gateway, Azure OpenAI, etc.)")
    llm_headers: dict[str, str] | None = Field(None, description="Extra HTTP headers for the LLM request (gateway auth, team ID, tracing, etc.)")


class PipelineResponse(BaseModel):
    selected_tool: str
    arguments: dict
    tool_result: dict | str | None
    llm_context: dict | list
    llm_response: str | None = None
    confidence: float
    timings: dict


# ---------------------------------------------------------------------------
# Multi-tenant tool registry
# ---------------------------------------------------------------------------


@dataclass
class ToolSetEntry:
    """One registered tool set."""
    tool_id: str
    tools: list[dict]
    tool_schemas: list[str]
    tool_names: list[str]
    group_key: str
    registered_at: float
    last_accessed_at: float
    query_count: int = 0


class ToolRegistry:
    """In-memory registry of named tool sets with LRU eviction."""

    def __init__(self, max_entries: int = 50):
        self._entries: OrderedDict[str, ToolSetEntry] = OrderedDict()
        self.max_entries = max_entries

    def register(self, tool_id: str, tools: list[dict], group_key: str) -> ToolSetEntry:
        tool_schemas = [json.dumps(t, separators=(",", ":")) for t in tools]
        tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
        now = time.time()
        entry = ToolSetEntry(
            tool_id=tool_id, tools=tools, tool_schemas=tool_schemas,
            tool_names=tool_names, group_key=group_key,
            registered_at=now, last_accessed_at=now,
        )
        if tool_id in self._entries:
            del self._entries[tool_id]
        self._entries[tool_id] = entry
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return entry

    def get(self, tool_id: str) -> ToolSetEntry | None:
        entry = self._entries.get(tool_id)
        if entry:
            entry.last_accessed_at = time.time()
            self._entries.move_to_end(tool_id)
        return entry

    def remove(self, tool_id: str) -> ToolSetEntry | None:
        return self._entries.pop(tool_id, None)

    def list_all(self) -> list[ToolSetEntry]:
        return list(self._entries.values())

    def __contains__(self, tool_id: str) -> bool:
        return tool_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)


def parse_tool_calls(raw_response: str) -> tuple[list[ToolSelection], float]:
    """Parse tool calls from model output.

    Returns (selections, confidence) where confidence is 1.0 for clean
    <tool_call> parses, 0.5 for regex fallback, 0.0 for no parse.
    """
    selections = []

    # Try structured <tool_call> blocks first
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, raw_response, re.DOTALL)

    if matches:
        for match in matches:
            try:
                data = json.loads(match)
                selections.append(ToolSelection(
                    tool_name=data.get("name", "unknown"),
                    arguments=data.get("arguments", {}),
                ))
            except json.JSONDecodeError:
                continue
        if selections:
            return selections, 1.0

    # Fallback: find JSON with "name" key
    json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}'
    for m in re.finditer(json_pattern, raw_response):
        try:
            data = json.loads(m.group(0))
            selections.append(ToolSelection(
                tool_name=data["name"],
                arguments=data.get("arguments", {}),
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    return selections, (0.5 if selections else 0.0)


# ---------------------------------------------------------------------------
# Demo mode helpers
# ---------------------------------------------------------------------------


def _gauss(mean, std):
    """Random gaussian with clamp to positive."""
    if isinstance(mean, dict):
        return max(0.1, random.gauss(mean.get("mean", 0), mean.get("std", 0)))
    return max(0.1, random.gauss(mean, std))


def _fuzzy_match(query, recordings):
    """Find best matching recording by keyword overlap."""
    query_lower = query.lower()
    best_score = 0
    best_rec = None
    for rec in recordings:
        score = sum(1 for kw in rec.get("keywords", []) if kw.lower() in query_lower)
        if score > best_score:
            best_score = score
            best_rec = rec
    return best_rec if best_score > 0 else None


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------


class ServerState:
    def __init__(self):
        self.model = None
        self.config = None
        self._loading: bool = False
        # Multi-tenant registry
        self.registry = ToolRegistry(max_entries=50)
        self._default_tool_id: str | None = None
        # Memory manager
        self.memory_manager = None
        # Server-side LLM config
        self.llm_config_store = None
        # Demo
        self.demo_mode: bool = False
        self.demo_recording: dict = {}
        self.demo_profiles: dict = {}
        self.demo_current_profile: str | None = None
        self._demo_query_counts: dict[str, int] = {}

    def load_model(self, config):
        from context_cache.context_cache import ContextCacheModel
        from context_cache.memory_manager import MemoryManager
        self.config = config
        self.model = ContextCacheModel(config)
        self.memory_manager = MemoryManager(max_gpu_cache_mb=config.cache.max_gpu_cache_mb)

    # --- Backward-compat properties for V1 endpoints ---

    @property
    def tools_loaded(self) -> bool:
        if self._default_tool_id is None:
            return False
        return self._default_tool_id in self.registry

    @property
    def current_tools(self) -> list[dict]:
        entry = self.registry.get(self._default_tool_id) if self._default_tool_id else None
        return entry.tools if entry else []

    @property
    def current_tool_schemas(self) -> list[str]:
        entry = self.registry.get(self._default_tool_id) if self._default_tool_id else None
        return entry.tool_schemas if entry else []

    @property
    def current_tool_names(self) -> list[str]:
        entry = self.registry.get(self._default_tool_id) if self._default_tool_id else None
        return entry.tool_names if entry else []

    @property
    def current_group_key(self) -> str | None:
        entry = self.registry.get(self._default_tool_id) if self._default_tool_id else None
        return entry.group_key if entry else None

    @property
    def _demo_query_count(self) -> int:
        return self._demo_query_counts.get(self._default_tool_id or "__default__", 0)

    @_demo_query_count.setter
    def _demo_query_count(self, value: int):
        self._demo_query_counts[self._default_tool_id or "__default__"] = value


state = ServerState()

_startup_args = {"config_path": None, "preload_tools": None, "demo": False, "auth": False}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    demo = _startup_args["demo"]
    config_path = _startup_args["config_path"]
    preload_tools = _startup_args["preload_tools"]

    # Initialize server-side LLM config
    from context_cache.llm_config import LLMConfigStore
    llm_config_path = _startup_args.get("llm_config")
    if llm_config_path and Path(llm_config_path).exists():
        state.llm_config_store = LLMConfigStore.from_json(llm_config_path)
        print(f"  LLM config: loaded from {llm_config_path}")
    else:
        state.llm_config_store = LLMConfigStore.from_env()
        if state.llm_config_store._default:
            print("  LLM config: loaded from environment (CONTEXTCACHE_LLM_*)")
        else:
            print("  LLM config: none (pass credentials per-request or use --llm-config)")

    if demo:
        state.demo_mode = True
        for name, path in [("recording", "demo_recording.json"), ("profiles", "demo_profiles.json")]:
            p = SERVE_DIR / path
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    if name == "recording":
                        state.demo_recording = json.load(f)
                    else:
                        state.demo_profiles = json.load(f)
        print("Demo mode: loaded recordings and profiles (no GPU needed)")
    elif config_path:
        from context_cache.cache_config import ContextCacheConfig
        state._loading = True
        config = ContextCacheConfig.from_yaml(config_path)
        state.config = config

        def _load():
            print("Loading model (this takes ~30s)...")
            state.load_model(config)
            print("Model loaded and ready.")
            if preload_tools and preload_tools.exists():
                _preload_tools(preload_tools)
            state._loading = False

        threading.Thread(target=_load, daemon=True).start()

    yield


def _preload_tools(tools_path):
    from context_cache.context_cache import ContextCacheModel
    print(f"Preloading tools from {tools_path}...")
    with open(tools_path, encoding="utf-8") as f:
        tools = json.load(f)

    tool_schemas, tool_names = [], []
    for tool in tools:
        schema = tool.get("schema", tool)
        tool_schemas.append(json.dumps(schema, separators=(",", ":")))
        tool_names.append(schema.get("function", {}).get("name", "unknown"))

    group_key = ContextCacheModel.compute_group_key(tool_schemas)
    if state.model.load_group_cache(group_key):
        print(f"  Loaded {len(tools)} tools from disk cache")
    else:
        print(f"  Compiling {len(tools)} tools...")
        t0 = time.perf_counter()
        state.model.generate_group_cached(context_texts=tool_schemas, user_query="hello", max_new_tokens=1)
        print(f"  Compiled in {(time.perf_counter() - t0) * 1000:.0f}ms")
        state.model.save_group_cache(group_key, tool_names=tool_names)

    state.registry.register("__default__", tools, group_key)
    state._default_tool_id = "__default__"
    print(f"  {len(tools)} tools ready")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ContextCache Server", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Auth + rate limiting middleware (conditionally added at startup)
_metrics_collector = None
_key_store = None


def _setup_auth():
    """Install auth middleware if --auth flag is set."""
    global _metrics_collector, _key_store
    from context_cache.middleware import (
        APIKeyStore, AuthRateLimitMiddleware, MetricsCollector, SlidingWindowRateLimiter,
    )

    _metrics_collector = MetricsCollector()
    rate_limiter = SlidingWindowRateLimiter(default_rpm=60)

    # Load API keys from file or env
    keys_path = ROOT / "api_keys.json"
    env_key = os.environ.get("CONTEXTCACHE_API_KEY")

    if keys_path.exists():
        _key_store = APIKeyStore.from_json(keys_path)
        print(f"  Auth: loaded {_key_store.num_keys} API key(s) from {keys_path}")
    elif env_key:
        _key_store = APIKeyStore.from_env_key(env_key)
        print("  Auth: using CONTEXTCACHE_API_KEY from environment")
    else:
        # Generate a temporary key for development
        import secrets
        temp_key = secrets.token_urlsafe(32)
        _key_store = APIKeyStore()
        _key_store.add_key(temp_key, name="dev-key", rate_limit=120)
        print(f"  Auth: generated dev key: {temp_key}")

    app.add_middleware(
        AuthRateLimitMiddleware,
        key_store=_key_store,
        rate_limiter=rate_limiter,
        metrics=_metrics_collector,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _compile_tool_set(tools: list[dict]) -> tuple[str, list[str], list[str], float, float, int, bool]:
    """Shared compilation logic. Returns (group_key, tool_schemas, tool_names, compile_ms, size_mb, prefix_tokens, from_disk)."""
    from context_cache.context_cache import ContextCacheModel

    tool_schemas = [json.dumps(t, separators=(",", ":")) for t in tools]
    tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
    group_key = ContextCacheModel.compute_group_key(tool_schemas)

    # Already in memory?
    if group_key in state.model._group_cache:
        cached_layers, prefix_len = state.model._group_cache[group_key]
        k0, v0 = cached_layers[0]
        per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
        size_mb = per_layer * len(cached_layers) / (1024 * 1024)
        return group_key, tool_schemas, tool_names, 0.0, size_mb, prefix_len, False

    # Try disk
    t0 = time.perf_counter()
    if state.model.load_group_cache(group_key):
        elapsed = (time.perf_counter() - t0) * 1000
        cached_layers, prefix_len = state.model._group_cache[group_key]
        k0, v0 = cached_layers[0]
        per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
        size_mb = per_layer * len(cached_layers) / (1024 * 1024)
        return group_key, tool_schemas, tool_names, elapsed, size_mb, prefix_len, True

    # Compile fresh
    t0 = time.perf_counter()
    state.model.generate_group_cached(context_texts=tool_schemas, user_query="hello", max_new_tokens=1)
    compile_ms = (time.perf_counter() - t0) * 1000
    if group_key not in state.model._group_cache:
        raise HTTPException(500, "Compilation failed")
    state.model.save_group_cache(group_key, tool_names=tool_names)

    cached_layers, prefix_len = state.model._group_cache[group_key]
    k0, v0 = cached_layers[0]
    per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
    size_mb = per_layer * len(cached_layers) / (1024 * 1024)

    # Track in memory manager
    if state.memory_manager:
        state.memory_manager.register(group_key, size_mb, tool_names)

    return group_key, tool_schemas, tool_names, compile_ms, size_mb, prefix_len, False


@app.post("/tools", response_model=ToolsResponse)
async def register_tools(req: ToolsRequest):
    if state.demo_mode:
        tp = state.demo_recording.get("timing_profiles", {})
        compile_cfg = tp.get("compile", {"mean": 1386, "std": 85})
        delay_ms = _gauss(compile_cfg["mean"], compile_cfg["std"])
        await asyncio.sleep(delay_ms / 1000)

        tool_schemas = [json.dumps(t, separators=(",", ":")) for t in req.tools]
        schema_blob = "".join(sorted(tool_schemas))
        group_key = "demo_" + hashlib.sha256(schema_blob.encode()).hexdigest()[:16]

        state.registry.register("__default__", req.tools, group_key)
        state._default_tool_id = "__default__"
        state._demo_query_counts["__default__"] = 0
        meta = state.demo_recording.get("metadata", {})

        return ToolsResponse(
            status="compiled", num_tools=len(req.tools),
            cache_hash=group_key[:16],
            compile_ms=round(delay_ms, 1),
            cache_size_mb=meta.get("cache_size_mb", 401.0),
            prefix_tokens=meta.get("prefix_tokens", 2851),
        )

    # Live mode
    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")

    group_key, tool_schemas, tool_names, compile_ms, size_mb, prefix_tokens, from_disk = \
        _compile_tool_set(req.tools)

    state.registry.register("__default__", req.tools, group_key)
    state._default_tool_id = "__default__"

    status = "loaded_from_disk" if from_disk else ("already_cached" if compile_ms == 0.0 else "compiled")
    return ToolsResponse(
        status=status, num_tools=len(tool_schemas),
        cache_hash=group_key[:16], compile_ms=round(compile_ms, 1),
        cache_size_mb=round(size_mb, 1), prefix_tokens=prefix_tokens,
        from_disk=from_disk,
    )


@app.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    if state.demo_mode:
        profile_key = state.demo_current_profile or "customer_service"
        recordings = state.demo_recording.get("recordings", {}).get(profile_key, [])
        match = _fuzzy_match(req.query, recordings)

        tp = state.demo_recording.get("timing_profiles", {})
        cached_tp = tp.get("cached", {})
        state._demo_query_count += 1
        is_hit = state._demo_query_count > 1

        if is_hit:
            link = _gauss(cached_tp.get("link_ms", {"mean": 2.0, "std": 0.13}), 0)
            prefill = _gauss(cached_tp.get("prefill_ms", {"mean": 112.0, "std": 11.4}), 0)
            ttft = link + prefill
        else:
            fp = tp.get("full_prefill", {"mean": 787, "std": 32})
            ttft = _gauss(fp["mean"], fp["std"])
            link, prefill = 0.0, ttft

        decode = _gauss(cached_tp.get("decode_ms", {"mean": 2100, "std": 300}), 0)
        await asyncio.sleep(min(ttft / 1000, 1.5))

        return QueryResponse(
            response=match["response"] if match else "<tool_call>\n{}\n</tool_call>",
            cache_hit=is_hit,
            timings={"cache_hit": is_hit, "link_ms": round(link, 1),
                     "prefill_query_ms": round(prefill, 1), "decode_ms": round(decode, 1),
                     "total_ms": round(ttft + decode, 1), "ttft_ms": round(ttft, 1)},
        )

    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")
    if not state.tools_loaded:
        raise HTTPException(400, "No tools registered. Call POST /tools first.")

    t0 = time.perf_counter()
    text, timings = state.model.generate_group_cached(
        context_texts=state.current_tool_schemas, user_query=req.query,
        max_new_tokens=req.max_new_tokens,
    )
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timings = {k: round(v, 1) if isinstance(v, float) else v for k, v in timings.items()}
    return QueryResponse(response=text.strip(), cache_hit=timings.get("cache_hit", False), timings=timings)


@app.post("/query/compare", response_model=CompareResponse)
async def compare_query(req: QueryRequest):
    if state.demo_mode:
        tp = state.demo_recording.get("timing_profiles", {})
        fp = tp.get("full_prefill", {"mean": 787, "std": 32})
        cached_tp = tp.get("cached", {})

        full_ttft = _gauss(fp["mean"], fp["std"])
        c_link = _gauss(cached_tp.get("link_ms", {"mean": 2.0, "std": 0.13}), 0)
        c_prefill = _gauss(cached_tp.get("prefill_ms", {"mean": 112.0, "std": 11.4}), 0)
        c_ttft = c_link + c_prefill

        await asyncio.sleep(min(full_ttft / 1000, 1.5))

        profile_key = state.demo_current_profile or "customer_service"
        recordings = state.demo_recording.get("recordings", {}).get(profile_key, [])
        match = _fuzzy_match(req.query, recordings)

        return CompareResponse(
            response=match["response"] if match else "<tool_call>\n{}\n</tool_call>",
            cached_timings={"ttft_ms": round(c_ttft, 1), "link_ms": round(c_link, 1), "prefill_ms": round(c_prefill, 1)},
            full_prefill_timings={"ttft_ms": round(full_ttft, 1)},
            speedup=round(full_ttft / max(c_ttft, 1), 1),
        )

    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")
    if not state.tools_loaded:
        raise HTTPException(400, "No tools registered.")

    # Run full prefill baseline (measures real TTFT without caching)
    _, full_timings = state.model.generate_full_prefill(
        context_texts=state.current_tool_schemas, user_query=req.query, max_new_tokens=req.max_new_tokens,
    )
    full_ttft = full_timings["prefill_ms"]

    # Run cached path
    t0 = time.perf_counter()
    text, cached_timings = state.model.generate_group_cached(
        context_texts=state.current_tool_schemas, user_query=req.query, max_new_tokens=req.max_new_tokens,
    )
    cached_timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    cached_ttft = cached_timings.get("link_ms", 0) + cached_timings.get("prefill_query_ms", 0)

    return CompareResponse(
        response=text.strip(),
        cached_timings={k: round(v, 1) if isinstance(v, float) else v for k, v in cached_timings.items()},
        full_prefill_timings={
            "ttft_ms": round(full_ttft, 1),
            "prompt_tokens": full_timings.get("prompt_tokens", 0),
        },
        speedup=round(full_ttft / max(cached_ttft, 0.1), 1),
    )


@app.get("/status")
def get_status():
    if state.demo_mode:
        meta = state.demo_recording.get("metadata", {})
        return {
            "tools_loaded": state.tools_loaded, "num_tools": len(state.current_tool_names),
            "tool_names": state.current_tool_names,
            "cache_hash": state.current_group_key[:16] if state.current_group_key else None,
            "cache_size_mb": meta.get("cache_size_mb", 401.0) if state.tools_loaded else 0.0,
            "prefix_tokens": meta.get("prefix_tokens", 2851) if state.tools_loaded else 0,
            "model_name": meta.get("model", "Qwen/Qwen3-8B (demo)"),
            "cache_info": {"groups_cached": 1 if state.tools_loaded else 0},
        }
    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")
    cache_info = state.model.group_cache_info
    size_mb, prefix_tokens = 0.0, 0
    if state.current_group_key and state.current_group_key in state.model._group_cache:
        cached_layers, prefix_tokens = state.model._group_cache[state.current_group_key]
        k0, v0 = cached_layers[0]
        per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
        size_mb = per_layer * len(cached_layers) / (1024 * 1024)
    return {
        "tools_loaded": state.tools_loaded, "num_tools": len(state.current_tool_schemas),
        "tool_names": state.current_tool_names,
        "cache_hash": state.current_group_key[:16] if state.current_group_key else None,
        "cache_size_mb": round(size_mb, 1), "prefix_tokens": prefix_tokens,
        "model_name": state.config.model.model_name if state.config else "not loaded",
        "cache_info": cache_info,
    }


@app.delete("/tools")
def clear_tools():
    if not state.demo_mode and state.model is None:
        raise HTTPException(503, "Model not loaded yet")
    if not state.demo_mode:
        state.model.clear_group_cache()
    entry = state.registry.get(state._default_tool_id) if state._default_tool_id else None
    old = len(entry.tool_schemas) if entry else 0
    state.registry.remove(state._default_tool_id) if state._default_tool_id else None
    state._default_tool_id = None
    state.demo_current_profile = None
    state._demo_query_counts.clear()
    return {"status": "cleared", "tools_removed": old}


@app.get("/health")
def health():
    if state.demo_mode:
        return {"status": "healthy", "model_loaded": True, "tools_loaded": state.tools_loaded, "mode": "demo"}
    return {"status": "healthy" if state.model else "loading",
            "model_loaded": state.model is not None, "tools_loaded": state.tools_loaded, "mode": "live"}


@app.get("/mode")
def get_mode():
    return {"mode": "demo" if state.demo_mode else "live"}


@app.get("/sample-tools")
def sample_tools():
    p = SERVE_DIR / "sample_tools.json"
    if not p.exists():
        raise HTTPException(404, "sample_tools.json not found")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@app.get("/profiles")
def list_profiles():
    result = []
    for key, prof in state.demo_profiles.get("profiles", {}).items():
        result.append({
            "id": key, "name": prof.get("name", key),
            "description": prof.get("description", ""),
            "num_tools": len(prof.get("tools", [])),
            "sample_queries": prof.get("sample_queries", []),
        })
    return result


@app.get("/profiles/{profile_id}/tools")
def get_profile_tools(profile_id: str):
    profiles = state.demo_profiles.get("profiles", {})
    if profile_id not in profiles:
        raise HTTPException(404, f"Profile '{profile_id}' not found")
    state.demo_current_profile = profile_id
    return profiles[profile_id].get("tools", [])


# ---------------------------------------------------------------------------
# V2 multi-tenant endpoints
# ---------------------------------------------------------------------------


@app.post("/v2/tools", response_model=ToolsResponseV2)
async def register_tools_v2(req: ToolsRequestV2):
    """Register a named tool set. Multiple tool sets coexist simultaneously."""
    if state.demo_mode:
        tp = state.demo_recording.get("timing_profiles", {})
        compile_cfg = tp.get("compile", {"mean": 1386, "std": 85})
        delay_ms = _gauss(compile_cfg["mean"], compile_cfg["std"])
        await asyncio.sleep(delay_ms / 1000)

        tool_schemas = [json.dumps(t, separators=(",", ":")) for t in req.tools]
        schema_blob = "".join(sorted(tool_schemas))
        group_key = "demo_" + hashlib.sha256(schema_blob.encode()).hexdigest()[:16]

        state.registry.register(req.tool_id, req.tools, group_key)
        state._demo_query_counts[req.tool_id] = 0
        meta = state.demo_recording.get("metadata", {})

        return ToolsResponseV2(
            status="compiled", tool_id=req.tool_id,
            num_tools=len(req.tools), cache_hash=group_key[:16],
            compile_ms=round(delay_ms, 1),
            cache_size_mb=meta.get("cache_size_mb", 401.0),
            prefix_tokens=meta.get("prefix_tokens", 2851),
        )

    # Live mode
    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")

    group_key, tool_schemas, tool_names, compile_ms, size_mb, prefix_tokens, from_disk = \
        _compile_tool_set(req.tools)

    state.registry.register(req.tool_id, req.tools, group_key)

    status = "loaded_from_disk" if from_disk else ("already_cached" if compile_ms == 0.0 else "compiled")
    return ToolsResponseV2(
        status=status, tool_id=req.tool_id,
        num_tools=len(tool_schemas), cache_hash=group_key[:16],
        compile_ms=round(compile_ms, 1), cache_size_mb=round(size_mb, 1),
        prefix_tokens=prefix_tokens, from_disk=from_disk,
    )


@app.post("/route", response_model=RouteResponse)
async def route_tool(req: RouteRequest):
    """Select the right tool for a query. Returns structured tool selection only.

    This is the primary endpoint for the routing layer. The small model
    picks the tool + extracts parameters. The actual execution happens
    in your enterprise model (Claude, OpenAI, etc).
    """
    entry = state.registry.get(req.tool_id)
    if entry is None:
        raise HTTPException(404, f"Tool set '{req.tool_id}' not registered. Call POST /v2/tools first.")

    if state.demo_mode:
        profile_key = req.tool_id if req.tool_id in state.demo_recording.get("recordings", {}) else (
            state.demo_current_profile or "customer_service"
        )
        recordings = state.demo_recording.get("recordings", {}).get(profile_key, [])
        match = _fuzzy_match(req.query, recordings)

        tp = state.demo_recording.get("timing_profiles", {})
        cached_tp = tp.get("cached", {})
        count = state._demo_query_counts.get(req.tool_id, 0) + 1
        state._demo_query_counts[req.tool_id] = count
        is_hit = count > 1

        if is_hit:
            link = _gauss(cached_tp.get("link_ms", {"mean": 2.0, "std": 0.13}), 0)
            prefill = _gauss(cached_tp.get("prefill_ms", {"mean": 112.0, "std": 11.4}), 0)
        else:
            fp = tp.get("full_prefill", {"mean": 787, "std": 32})
            link, prefill = 0.0, _gauss(fp["mean"], fp["std"])

        decode = _gauss(cached_tp.get("decode_ms", {"mean": 2100, "std": 300}), 0)
        total = link + prefill + decode
        await asyncio.sleep(min((link + prefill) / 1000, 1.5))

        raw = match["response"] if match else '<tool_call>\n{"name": "unknown", "arguments": {}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)

        return RouteResponse(
            tool_id=req.tool_id, selections=selections,
            raw_response=raw, confidence=confidence,
            timings={"cache_hit": is_hit, "link_ms": round(link, 1),
                     "prefill_query_ms": round(prefill, 1), "decode_ms": round(decode, 1),
                     "total_ms": round(total, 1)},
        )

    # Live mode
    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")

    entry.query_count += 1
    t0 = time.perf_counter()
    raw_text, timings = state.model.generate_group_cached(
        context_texts=entry.tool_schemas, user_query=req.query,
        max_new_tokens=req.max_new_tokens,
    )
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timings = {k: round(v, 1) if isinstance(v, float) else v for k, v in timings.items()}

    selections, confidence = parse_tool_calls(raw_text.strip())

    return RouteResponse(
        tool_id=req.tool_id, selections=selections,
        raw_response=raw_text.strip(), confidence=confidence,
        timings=timings,
    )


@app.post("/v2/query", response_model=QueryResponseV2)
async def run_query_v2(req: QueryRequestV2):
    """Run a query against a named tool set. Returns raw model output."""
    entry = state.registry.get(req.tool_id)
    if entry is None:
        raise HTTPException(404, f"Tool set '{req.tool_id}' not registered.")

    if state.demo_mode:
        profile_key = req.tool_id if req.tool_id in state.demo_recording.get("recordings", {}) else (
            state.demo_current_profile or "customer_service"
        )
        recordings = state.demo_recording.get("recordings", {}).get(profile_key, [])
        match = _fuzzy_match(req.query, recordings)

        tp = state.demo_recording.get("timing_profiles", {})
        cached_tp = tp.get("cached", {})
        count = state._demo_query_counts.get(req.tool_id, 0) + 1
        state._demo_query_counts[req.tool_id] = count
        is_hit = count > 1

        if is_hit:
            link = _gauss(cached_tp.get("link_ms", {"mean": 2.0, "std": 0.13}), 0)
            prefill = _gauss(cached_tp.get("prefill_ms", {"mean": 112.0, "std": 11.4}), 0)
            ttft = link + prefill
        else:
            fp = tp.get("full_prefill", {"mean": 787, "std": 32})
            ttft = _gauss(fp["mean"], fp["std"])
            link, prefill = 0.0, ttft

        decode = _gauss(cached_tp.get("decode_ms", {"mean": 2100, "std": 300}), 0)
        await asyncio.sleep(min(ttft / 1000, 1.5))

        return QueryResponseV2(
            response=match["response"] if match else "<tool_call>\n{}\n</tool_call>",
            tool_id=req.tool_id, cache_hit=is_hit,
            timings={"cache_hit": is_hit, "link_ms": round(link, 1),
                     "prefill_query_ms": round(prefill, 1), "decode_ms": round(decode, 1),
                     "total_ms": round(ttft + decode, 1)},
        )

    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")

    entry.query_count += 1
    t0 = time.perf_counter()
    text, timings = state.model.generate_group_cached(
        context_texts=entry.tool_schemas, user_query=req.query,
        max_new_tokens=req.max_new_tokens,
    )
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timings = {k: round(v, 1) if isinstance(v, float) else v for k, v in timings.items()}

    return QueryResponseV2(
        response=text.strip(), tool_id=req.tool_id,
        cache_hit=timings.get("cache_hit", False), timings=timings,
    )


@app.get("/v2/registry")
def get_registry():
    """List all registered tool sets and their status."""
    entries = []
    for entry in state.registry.list_all():
        info = {
            "tool_id": entry.tool_id,
            "num_tools": len(entry.tool_names),
            "tool_names": entry.tool_names[:10],  # first 10 for brevity
            "total_tool_count": len(entry.tool_names),
            "group_key": entry.group_key[:16],
            "registered_at": entry.registered_at,
            "last_accessed_at": entry.last_accessed_at,
            "query_count": entry.query_count,
            "cache_in_memory": (
                entry.group_key in state.model._group_cache
                if state.model else True
            ),
        }
        entries.append(info)

    total_mb = 0.0
    if state.model:
        for key, (layers, _) in state.model._group_cache.items():
            k0, v0 = layers[0]
            per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
            total_mb += per_layer * len(layers) / (1024 * 1024)

    return {
        "num_tool_sets": len(state.registry),
        "tool_sets": entries,
        "total_cache_memory_mb": round(total_mb, 1),
    }


@app.delete("/v2/tools/{tool_id}")
def remove_tool_set(tool_id: str):
    """Remove a specific tool set from the registry."""
    entry = state.registry.remove(tool_id)
    if entry is None:
        raise HTTPException(404, f"Tool set '{tool_id}' not found")

    # Only evict KV cache if no other tool_id shares the same group_key
    shared = any(e.group_key == entry.group_key for e in state.registry.list_all())
    evicted_cache = False
    if not shared and not state.demo_mode and state.model:
        state.model._group_cache.pop(entry.group_key, None)
        evicted_cache = True

    state._demo_query_counts.pop(tool_id, None)
    return {
        "status": "removed", "tool_id": tool_id,
        "tools_removed": len(entry.tool_names),
        "cache_evicted": evicted_cache,
    }


# ---------------------------------------------------------------------------
# Pipeline endpoint
# ---------------------------------------------------------------------------


def _mock_tool_result(tool_name: str, arguments: dict) -> dict:
    """Generate a mock tool result for demo/testing."""
    return {
        "tool": tool_name,
        "status": "success",
        "data": {
            "message": f"Mock result from {tool_name}",
            "arguments_received": arguments,
        },
    }


def _format_for_claude(tool_name: str, arguments: dict, tool_result: dict, query: str) -> list[dict]:
    """Format tool result as Claude API messages (tool_use + tool_result pattern)."""
    return [
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"toolu_{hashlib.sha256(tool_name.encode()).hexdigest()[:12]}",
                    "name": tool_name,
                    "input": arguments,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"toolu_{hashlib.sha256(tool_name.encode()).hexdigest()[:12]}",
                    "content": json.dumps(tool_result),
                }
            ],
        },
    ]


def _format_for_openai(tool_name: str, arguments: dict, tool_result: dict, query: str) -> list[dict]:
    """Format tool result as OpenAI API messages (function call pattern)."""
    call_id = f"call_{hashlib.sha256(tool_name.encode()).hexdigest()[:12]}"
    return [
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(tool_result),
        },
    ]


@app.post("/v2/pipeline", response_model=PipelineResponse)
async def run_pipeline(req: PipelineRequest):
    """Full pipeline: route query → execute tool → format for enterprise LLM.

    Steps:
      1. Route: Use cached model to select the right tool
      2. Execute: Mock result or call external URL
      3. Format: Package as Claude/OpenAI message format

    The llm_context is ready to drop into your Anthropic or OpenAI SDK call.
    """
    entry = state.registry.get(req.tool_id)
    if entry is None:
        raise HTTPException(404, f"Tool set '{req.tool_id}' not registered.")

    pipeline_timings = {}

    # Step 1: Route
    if state.demo_mode:
        profile_key = req.tool_id if req.tool_id in state.demo_recording.get("recordings", {}) else (
            state.demo_current_profile or "customer_service"
        )
        recordings = state.demo_recording.get("recordings", {}).get(profile_key, [])
        match = _fuzzy_match(req.query, recordings)
        raw = match["response"] if match else '<tool_call>\n{"name": "unknown", "arguments": {}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        pipeline_timings["route_ms"] = round(random.gauss(150, 20), 1)
    else:
        if state.model is None:
            raise HTTPException(503, "Model not loaded yet")
        entry.query_count += 1

        t0 = time.perf_counter()
        raw_text, route_timings = state.model.generate_group_cached(
            context_texts=entry.tool_schemas, user_query=req.query,
            max_new_tokens=req.max_new_tokens,
        )
        pipeline_timings["route_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        pipeline_timings["route_details"] = {
            k: round(v, 1) if isinstance(v, float) else v for k, v in route_timings.items()
        }
        selections, confidence = parse_tool_calls(raw_text.strip())

    tool_name = selections[0].tool_name if selections else "unknown"
    arguments = selections[0].arguments if selections else {}

    # Step 2: Execute
    t0 = time.perf_counter()
    if req.tool_executor == "mock":
        tool_result = _mock_tool_result(tool_name, arguments)
    elif req.tool_executor.startswith("http"):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    req.tool_executor,
                    json={"tool": tool_name, "arguments": arguments},
                )
                tool_result = resp.json()
        except Exception as e:
            tool_result = {"error": str(e), "tool": tool_name}
    else:
        tool_result = _mock_tool_result(tool_name, arguments)

    pipeline_timings["execute_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Step 3: Format for LLM (and optionally call it)
    #
    # LLM credential resolution order:
    #   1. Request-level params (backward compatible)
    #   2. Server-side per-tool_id config
    #   3. Server-side global default
    llm_response = None

    # Resolve LLM config
    llm_api_key = req.llm_api_key
    llm_provider = req.llm_format
    llm_model = req.llm_model
    llm_base_url = req.llm_base_url
    llm_headers = req.llm_headers
    llm_system_prompt = req.llm_system_prompt

    if not llm_api_key and state.llm_config_store:
        server_config = state.llm_config_store.resolve(req.tool_id)
        if server_config:
            llm_api_key = server_config.api_key
            llm_provider = server_config.provider
            llm_model = llm_model or server_config.model
            llm_base_url = llm_base_url or server_config.base_url
            llm_headers = llm_headers or (server_config.extra_headers or None)
            llm_system_prompt = llm_system_prompt or server_config.system_prompt

    if llm_api_key and llm_provider in ("claude", "openai"):
        # Full LLM call — format + send to enterprise LLM
        from context_cache.llm_adapter import get_llm_adapter
        adapter = get_llm_adapter(
            llm_provider,
            api_key=llm_api_key,
            model=llm_model,
            base_url=llm_base_url,
            extra_headers=llm_headers,
        )
        llm_context = adapter.format_messages(
            tool_name, arguments, tool_result, req.query, system_prompt=llm_system_prompt,
        )
        try:
            t0 = time.perf_counter()
            result = await adapter.complete(
                llm_context, system_prompt=llm_system_prompt,
            )
            pipeline_timings["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            pipeline_timings["llm_model"] = result.model
            pipeline_timings["llm_usage"] = result.usage
            llm_response = result.content
        except Exception as e:
            pipeline_timings["llm_error"] = str(e)
            llm_response = None
    else:
        # Format only — no API call
        if req.llm_format == "claude":
            llm_context = _format_for_claude(tool_name, arguments, tool_result, req.query)
        elif req.llm_format == "openai":
            llm_context = _format_for_openai(tool_name, arguments, tool_result, req.query)
        else:
            llm_context = {"tool": tool_name, "arguments": arguments, "result": tool_result}

    pipeline_timings["total_ms"] = round(
        pipeline_timings.get("route_ms", 0)
        + pipeline_timings.get("execute_ms", 0)
        + pipeline_timings.get("llm_ms", 0),
        1,
    )
    pipeline_timings["cache_hit"] = pipeline_timings.get("route_details", {}).get("cache_hit", True)

    return PipelineResponse(
        selected_tool=tool_name,
        arguments=arguments,
        tool_result=tool_result,
        llm_context=llm_context,
        llm_response=llm_response,
        confidence=confidence,
        timings=pipeline_timings,
    )


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------


@app.get("/admin/metrics")
def get_metrics():
    """Request metrics per API key."""
    if _metrics_collector is None:
        return {"auth_enabled": False, "message": "Auth not enabled. Start with --auth flag."}
    return _metrics_collector.get_metrics()


@app.get("/admin/memory")
def get_memory():
    """GPU cache memory status."""
    if state.memory_manager:
        return state.memory_manager.get_status()
    # Fallback: compute from raw group cache
    if state.model:
        total_mb = 0.0
        for key, (layers, _) in state.model._group_cache.items():
            k0, v0 = layers[0]
            per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
            total_mb += per_layer * len(layers) / (1024 * 1024)
        return {"total_cache_mb": round(total_mb, 1), "num_entries": len(state.model._group_cache)}
    return {"total_cache_mb": 0, "num_entries": 0}


@app.post("/admin/evict")
def evict_caches():
    """Force evict LRU caches to free GPU memory."""
    if not state.memory_manager or not state.model:
        return {"evicted": 0}

    candidates = state.memory_manager.eviction_candidates()
    evicted = 0
    for key in candidates:
        state.model._group_cache.pop(key, None)
        state.memory_manager.remove(key)
        evicted += 1

    return {
        "evicted": evicted,
        "remaining_mb": round(state.memory_manager.total_mb, 1),
    }


# ---------------------------------------------------------------------------
# LLM config admin endpoints
# ---------------------------------------------------------------------------


class LLMConfigRequest(BaseModel):
    provider: str = Field("claude", description="LLM provider: 'claude' or 'openai'")
    api_key: str = Field(..., description="API key for the LLM provider")
    model: str | None = Field(None, description="Model name (e.g. 'claude-sonnet-4-20250514', 'gpt-4o')")
    base_url: str | None = Field(None, description="Custom endpoint URL (enterprise gateway)")
    extra_headers: dict[str, str] | None = Field(None, description="Additional HTTP headers")
    system_prompt: str | None = Field(None, description="Default system prompt")


@app.post("/admin/llm-config/{tool_id}")
def set_llm_config(tool_id: str, req: LLMConfigRequest):
    """Set server-side LLM config for a tool_id. Keeps API keys off the wire."""
    from context_cache.llm_config import LLMConfig
    if state.llm_config_store is None:
        from context_cache.llm_config import LLMConfigStore
        state.llm_config_store = LLMConfigStore()

    config = LLMConfig(
        provider=req.provider,
        api_key=req.api_key,
        model=req.model,
        base_url=req.base_url,
        extra_headers=req.extra_headers or {},
        system_prompt=req.system_prompt,
    )
    if tool_id == "_default":
        state.llm_config_store.set_default(config)
    else:
        state.llm_config_store.set(tool_id, config)

    return {"status": "configured", "tool_id": tool_id, "config": config.to_safe_dict()}


@app.get("/admin/llm-config")
def list_llm_configs():
    """List all server-side LLM configs (API keys masked)."""
    if state.llm_config_store is None:
        return {"configs": {}, "message": "No LLM configs set."}
    return {"configs": state.llm_config_store.list_configured()}


@app.delete("/admin/llm-config/{tool_id}")
def remove_llm_config(tool_id: str):
    """Remove LLM config for a tool_id."""
    if state.llm_config_store is None:
        raise HTTPException(404, "No LLM configs configured.")
    if tool_id == "_default":
        state.llm_config_store._default = None
    else:
        state.llm_config_store.remove(tool_id)
    return {"status": "removed", "tool_id": tool_id}


# Static files — must be last
STATIC_DIR = SERVE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main():
    parser = argparse.ArgumentParser(description="ContextCache API Server")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "context_cache_config.yaml")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument("--preload-tools", type=Path, default=None)
    parser.add_argument("--demo", action="store_true", help="Demo mode (no GPU, simulated responses)")
    parser.add_argument("--auth", action="store_true", help="Enable API key authentication")
    parser.add_argument("--llm-config", type=Path, default=None, help="Path to LLM config JSON (server-side credentials)")
    args = parser.parse_args()

    _startup_args["config_path"] = args.config if not args.demo else None
    _startup_args["preload_tools"] = args.preload_tools
    _startup_args["demo"] = args.demo
    _startup_args["auth"] = args.auth
    _startup_args["llm_config"] = args.llm_config

    mode = "DEMO" if args.demo else "LIVE"
    print(f"\nContextCache Server [{mode}]")
    if not args.demo:
        print(f"  Config: {args.config}")

    if args.auth:
        _setup_auth()
    else:
        print("  Auth:   disabled (use --auth to enable)")

    print(f"  URL:    http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
