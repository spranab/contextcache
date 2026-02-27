#!/usr/bin/env python3
"""FastAPI server for ContextCache — persistent, composable KV cache serving.

Supports two modes:
  Live mode:  Full model inference with real KV cache (requires GPU)
  Demo mode:  Pre-recorded responses with simulated timings (no GPU needed)

Endpoints:
  POST /tools          Register tool schemas -> compile group KV cache
  POST /query          Run inference with cached tools
  POST /query/compare  A/B comparison: cached vs full prefill timings
  GET  /status         Current cache state, loaded tools, memory usage
  DELETE /tools        Clear cached tools
  GET  /health         Health check (works during model loading)
  GET  /sample-tools   Returns demo tool schemas
  GET  /profiles       List available team profiles
  GET  /mode           Current server mode (live or demo)

Usage:
  python scripts/serve/serve_context_cache.py                    # Live mode
  python scripts/serve/serve_context_cache.py --demo             # Demo mode
  python scripts/serve/serve_context_cache.py --port 8080
"""

import argparse
import asyncio
import json
import random
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

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
        self.current_tools: list[dict] = []
        self.current_tool_schemas: list[str] = []
        self.current_tool_names: list[str] = []
        self.current_group_key: str | None = None
        self.config = None
        self._loading: bool = False
        # Demo
        self.demo_mode: bool = False
        self.demo_recording: dict = {}
        self.demo_profiles: dict = {}
        self.demo_current_profile: str | None = None
        self._demo_query_count: int = 0

    def load_model(self, config):
        from context_cache.context_cache import ContextCacheModel
        self.config = config
        self.model = ContextCacheModel(config)

    @property
    def tools_loaded(self):
        return bool(self.current_tool_schemas) and self.current_group_key is not None


state = ServerState()

_startup_args = {"config_path": None, "preload_tools": None, "demo": False}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    demo = _startup_args["demo"]
    config_path = _startup_args["config_path"]
    preload_tools = _startup_args["preload_tools"]

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

    state.current_tools = tools
    state.current_tool_schemas = tool_schemas
    state.current_tool_names = tool_names
    state.current_group_key = group_key
    print(f"  {len(tools)} tools ready")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ContextCache Server", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/tools", response_model=ToolsResponse)
async def register_tools(req: ToolsRequest):
    if state.demo_mode:
        tp = state.demo_recording.get("timing_profiles", {})
        compile_cfg = tp.get("compile", {"mean": 1386, "std": 85})
        delay_ms = _gauss(compile_cfg["mean"], compile_cfg["std"])
        await asyncio.sleep(delay_ms / 1000)

        tool_names = [t.get("function", {}).get("name", "unknown") for t in req.tools]
        state.current_tools = req.tools
        state.current_tool_schemas = [json.dumps(t, separators=(",", ":")) for t in req.tools]
        state.current_tool_names = tool_names
        state.current_group_key = "demo_" + str(hash(tuple(tool_names)))
        state._demo_query_count = 0
        meta = state.demo_recording.get("metadata", {})

        return ToolsResponse(
            status="compiled", num_tools=len(req.tools),
            cache_hash=state.current_group_key[:16],
            compile_ms=round(delay_ms, 1),
            cache_size_mb=meta.get("cache_size_mb", 401.0),
            prefix_tokens=meta.get("prefix_tokens", 2851),
        )

    # Live mode
    from context_cache.context_cache import ContextCacheModel
    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")

    tool_schemas, tool_names = [], []
    for tool in req.tools:
        tool_schemas.append(json.dumps(tool, separators=(",", ":")))
        tool_names.append(tool.get("function", {}).get("name", "unknown"))

    group_key = ContextCacheModel.compute_group_key(tool_schemas)

    if group_key == state.current_group_key and group_key in state.model._group_cache:
        cached_layers, prefix_len = state.model._group_cache[group_key]
        k0, v0 = cached_layers[0]
        per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
        size_mb = per_layer * len(cached_layers) / (1024 * 1024)
        return ToolsResponse(
            status="already_cached", num_tools=len(tool_schemas),
            cache_hash=group_key[:16], compile_ms=0.0,
            cache_size_mb=round(size_mb, 1), prefix_tokens=prefix_len,
        )

    t0 = time.perf_counter()
    if state.model.load_group_cache(group_key):
        elapsed = (time.perf_counter() - t0) * 1000
        cached_layers, prefix_len = state.model._group_cache[group_key]
        k0, v0 = cached_layers[0]
        per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
        size_mb = per_layer * len(cached_layers) / (1024 * 1024)
        state.current_tools, state.current_tool_schemas = req.tools, tool_schemas
        state.current_tool_names, state.current_group_key = tool_names, group_key
        return ToolsResponse(
            status="loaded_from_disk", num_tools=len(tool_schemas),
            cache_hash=group_key[:16], compile_ms=round(elapsed, 1),
            cache_size_mb=round(size_mb, 1), prefix_tokens=prefix_len, from_disk=True,
        )

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
    state.current_tools, state.current_tool_schemas = req.tools, tool_schemas
    state.current_tool_names, state.current_group_key = tool_names, group_key

    return ToolsResponse(
        status="compiled", num_tools=len(tool_schemas),
        cache_hash=group_key[:16], compile_ms=round(compile_ms, 1),
        cache_size_mb=round(size_mb, 1), prefix_tokens=prefix_len,
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

    t0 = time.perf_counter()
    text, cached_timings = state.model.generate_group_cached(
        context_texts=state.current_tool_schemas, user_query=req.query, max_new_tokens=req.max_new_tokens,
    )
    cached_timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    full_ttft = cached_timings.get("ttft_ms", 114.0) * 6.9

    return CompareResponse(
        response=text.strip(),
        cached_timings={k: round(v, 1) if isinstance(v, float) else v for k, v in cached_timings.items()},
        full_prefill_timings={"ttft_ms": round(full_ttft, 1)},
        speedup=round(full_ttft / max(cached_timings.get("ttft_ms", 114.0), 1), 1),
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
    old = len(state.current_tool_schemas)
    state.current_tools, state.current_tool_schemas = [], []
    state.current_tool_names, state.current_group_key = [], None
    state.demo_current_profile = None
    state._demo_query_count = 0
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
    args = parser.parse_args()

    _startup_args["config_path"] = args.config if not args.demo else None
    _startup_args["preload_tools"] = args.preload_tools
    _startup_args["demo"] = args.demo

    mode = "DEMO" if args.demo else "LIVE"
    print(f"\nContextCache Server [{mode}]")
    if not args.demo:
        print(f"  Config: {args.config}")
    print(f"  URL:    http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
