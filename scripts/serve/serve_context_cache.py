#!/usr/bin/env python3
"""FastAPI server for ContextCache — persistent, composable KV cache serving.

Endpoints:
  POST /tools     Register tool schemas → compile group KV cache
  POST /query     Run inference with cached tools (no schemas needed)
  GET  /status    Current cache state, loaded tools, memory usage
  DELETE /tools   Clear cached tools
  GET  /health    Health check (works during model loading)
  GET  /sample-tools  Returns demo tool schemas

Web UI:
  GET /           Browser-based UI for tool registration and querying

Deployment flow:
  1. On startup (or tool change), POST /tools with the full tool catalog
  2. Server compiles group KV cache (~1.4s for 20 tools), persists to disk
  3. All subsequent POST /query requests use cached KV — no tool schemas needed
  4. TTFT drops from ~787ms to ~114ms (6.9x speedup)

Usage:
  python scripts/serve/serve_context_cache.py
  python scripts/serve/serve_context_cache.py --port 8080
  python scripts/serve/serve_context_cache.py --config configs/context_cache_config.yaml

  # Or use the launcher:
  python scripts/serve/launch.py
"""

import argparse
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
SERVE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from context_cache.cache_config import ContextCacheConfig
from context_cache.context_cache import ContextCacheModel

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ToolsRequest(BaseModel):
    """Register tool schemas for caching."""
    tools: list[dict] = Field(
        ...,
        description=(
            "List of tool schemas in OpenAI function-calling format. "
            'Each should have {"type": "function", "function": {"name": ..., "parameters": ...}}'
        ),
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
    """Run inference using cached tools."""
    query: str = Field(..., description="User query text")
    max_new_tokens: int = Field(256, ge=1, le=2048)


class QueryResponse(BaseModel):
    response: str
    cache_hit: bool
    timings: dict


class StatusResponse(BaseModel):
    tools_loaded: bool
    num_tools: int
    tool_names: list[str]
    cache_hash: str | None
    cache_size_mb: float
    prefix_tokens: int
    model_name: str
    cache_info: dict


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------


class ServerState:
    """Holds the model and current tool state."""

    def __init__(self):
        self.model: ContextCacheModel | None = None
        self.current_tools: list[dict] = []
        self.current_tool_schemas: list[str] = []  # JSON strings
        self.current_tool_names: list[str] = []
        self.current_group_key: str | None = None
        self.config: ContextCacheConfig | None = None
        self._loading: bool = False

    def load_model(self, config: ContextCacheConfig):
        self.config = config
        self.model = ContextCacheModel(config)

    @property
    def tools_loaded(self) -> bool:
        return bool(self.current_tool_schemas) and self.current_group_key is not None


state = ServerState()

# ---------------------------------------------------------------------------
# Startup arguments (set from main() before uvicorn.run)
# ---------------------------------------------------------------------------

_startup_args = {
    "config_path": None,
    "preload_tools": None,
}


# ---------------------------------------------------------------------------
# Lifespan — load model in background thread
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model in a background thread so /health responds immediately."""
    config_path = _startup_args["config_path"]
    preload_tools = _startup_args["preload_tools"]

    if config_path:
        state._loading = True
        config = ContextCacheConfig.from_yaml(config_path)
        state.config = config

        def _load():
            print("Loading model (this takes ~30s)...")
            state.load_model(config)
            print("Model loaded and ready.")

            # Optionally preload tools
            if preload_tools and preload_tools.exists():
                _preload_tools(preload_tools)

            state._loading = False

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    yield


def _preload_tools(tools_path: Path):
    """Load and compile tools from a JSON file."""
    print(f"Preloading tools from {tools_path}...")
    with open(tools_path, encoding="utf-8") as f:
        tools = json.load(f)

    tool_schemas = []
    tool_names = []
    for tool in tools:
        schema = tool.get("schema", tool)
        schema_text = json.dumps(schema, separators=(",", ":"))
        tool_schemas.append(schema_text)
        name = schema.get("function", {}).get("name", tool.get("tool_id", "unknown"))
        tool_names.append(name)

    group_key = ContextCacheModel.compute_group_key(tool_schemas)

    if state.model.load_group_cache(group_key):
        print(f"  Loaded {len(tools)} tools from disk cache")
    else:
        print(f"  Compiling {len(tools)} tools...")
        t0 = time.perf_counter()
        state.model.generate_group_cached(
            context_texts=tool_schemas,
            user_query="hello",
            max_new_tokens=1,
        )
        compile_ms = (time.perf_counter() - t0) * 1000
        state.model.save_group_cache(group_key, tool_names=tool_names)
        print(f"  Compiled and saved in {compile_ms:.0f}ms")

    state.current_tools = tools
    state.current_tool_schemas = tool_schemas
    state.current_tool_names = tool_names
    state.current_group_key = group_key
    print(f"  {len(tools)} tools ready to serve")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ContextCache Server",
    description="Persistent, composable KV cache for tool-augmented LLM inference",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/tools", response_model=ToolsResponse)
def register_tools(req: ToolsRequest):
    """Register tool schemas and compile group KV cache.

    This triggers a one-time compilation of the tool set's KV cache.
    The cache is persisted to disk and loaded into GPU memory.
    All subsequent /query requests will use this cached KV.
    """
    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")

    # Convert tool dicts to compact JSON strings (the cache key format)
    tool_schemas = []
    tool_names = []
    for tool in req.tools:
        schema_text = json.dumps(tool, separators=(",", ":"))
        tool_schemas.append(schema_text)
        name = tool.get("function", {}).get("name", "unknown")
        tool_names.append(name)

    # Compute group key
    group_key = ContextCacheModel.compute_group_key(tool_schemas)

    # Check if already cached in memory
    if group_key == state.current_group_key and group_key in state.model._group_cache:
        cached_layers, prefix_len = state.model._group_cache[group_key]
        k0, v0 = cached_layers[0]
        per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
        size_mb = per_layer * len(cached_layers) / (1024 * 1024)
        return ToolsResponse(
            status="already_cached",
            num_tools=len(tool_schemas),
            cache_hash=group_key[:16],
            compile_ms=0.0,
            cache_size_mb=round(size_mb, 1),
            prefix_tokens=prefix_len,
            from_disk=False,
        )

    # Check disk cache
    t0 = time.perf_counter()
    if state.model.load_group_cache(group_key):
        elapsed = (time.perf_counter() - t0) * 1000
        cached_layers, prefix_len = state.model._group_cache[group_key]
        k0, v0 = cached_layers[0]
        per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
        size_mb = per_layer * len(cached_layers) / (1024 * 1024)

        state.current_tools = req.tools
        state.current_tool_schemas = tool_schemas
        state.current_tool_names = tool_names
        state.current_group_key = group_key

        return ToolsResponse(
            status="loaded_from_disk",
            num_tools=len(tool_schemas),
            cache_hash=group_key[:16],
            compile_ms=round(elapsed, 1),
            cache_size_mb=round(size_mb, 1),
            prefix_tokens=prefix_len,
            from_disk=True,
        )

    # Cache miss — need to compile. Run a dummy query to trigger cache population.
    t0 = time.perf_counter()
    _text, _timings = state.model.generate_group_cached(
        context_texts=tool_schemas,
        user_query="hello",
        max_new_tokens=1,
    )
    compile_ms = (time.perf_counter() - t0) * 1000

    # Verify it's now cached
    if group_key not in state.model._group_cache:
        raise HTTPException(500, "Compilation failed — group cache not populated")

    # Persist to disk
    state.model.save_group_cache(group_key, tool_names=tool_names)

    # Compute size
    cached_layers, prefix_len = state.model._group_cache[group_key]
    k0, v0 = cached_layers[0]
    per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
    size_mb = per_layer * len(cached_layers) / (1024 * 1024)

    # Update state
    state.current_tools = req.tools
    state.current_tool_schemas = tool_schemas
    state.current_tool_names = tool_names
    state.current_group_key = group_key

    return ToolsResponse(
        status="compiled",
        num_tools=len(tool_schemas),
        cache_hash=group_key[:16],
        compile_ms=round(compile_ms, 1),
        cache_size_mb=round(size_mb, 1),
        prefix_tokens=prefix_len,
    )


@app.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    """Run inference using cached tool KV states.

    The client only sends the query text — tool schemas are already cached.
    This saves both token costs and latency.
    """
    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")

    if not state.tools_loaded:
        raise HTTPException(
            400,
            "No tools registered. Call POST /tools first to register and cache tool schemas.",
        )

    t0 = time.perf_counter()
    response_text, timings = state.model.generate_group_cached(
        context_texts=state.current_tool_schemas,
        user_query=req.query,
        max_new_tokens=req.max_new_tokens,
    )
    total_ms = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = round(total_ms, 1)
    # Round all timing values
    timings = {k: round(v, 1) if isinstance(v, float) else v for k, v in timings.items()}

    return QueryResponse(
        response=response_text.strip(),
        cache_hit=timings.get("cache_hit", False),
        timings=timings,
    )


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Get current server state: loaded tools, cache info, memory usage."""
    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")

    cache_info = state.model.group_cache_info

    # Compute current cache size
    size_mb = 0.0
    prefix_tokens = 0
    if state.current_group_key and state.current_group_key in state.model._group_cache:
        cached_layers, prefix_tokens = state.model._group_cache[state.current_group_key]
        k0, v0 = cached_layers[0]
        per_layer = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
        size_mb = per_layer * len(cached_layers) / (1024 * 1024)

    return StatusResponse(
        tools_loaded=state.tools_loaded,
        num_tools=len(state.current_tool_schemas),
        tool_names=state.current_tool_names,
        cache_hash=state.current_group_key[:16] if state.current_group_key else None,
        cache_size_mb=round(size_mb, 1),
        prefix_tokens=prefix_tokens,
        model_name=state.config.model.model_name if state.config else "not loaded",
        cache_info=cache_info,
    )


@app.delete("/tools")
def clear_tools():
    """Clear the current tool cache (both in-memory and on-disk)."""
    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")

    state.model.clear_group_cache()
    old_count = len(state.current_tool_schemas)

    state.current_tools = []
    state.current_tool_schemas = []
    state.current_tool_names = []
    state.current_group_key = None

    return {"status": "cleared", "tools_removed": old_count}


@app.get("/health")
def health():
    """Health check — works during model loading."""
    return {
        "status": "healthy" if state.model is not None else "loading",
        "model_loaded": state.model is not None,
        "tools_loaded": state.tools_loaded,
    }


@app.get("/sample-tools")
def sample_tools():
    """Return demo tool schemas for the UI."""
    sample_path = SERVE_DIR / "sample_tools.json"
    if not sample_path.exists():
        raise HTTPException(404, "sample_tools.json not found")
    with open(sample_path, encoding="utf-8") as f:
        return json.load(f)


# Serve the web UI — must be mounted LAST so API routes take priority
STATIC_DIR = SERVE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ContextCache API Server")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "context_cache_config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument(
        "--preload-tools", type=Path, default=None,
        help="Path to a tools JSON file to preload on startup",
    )
    args = parser.parse_args()

    # Store args for lifespan to pick up
    _startup_args["config_path"] = args.config
    _startup_args["preload_tools"] = args.preload_tools

    print(f"\nContextCache Server")
    print(f"  Config: {args.config}")
    print(f"  URL:    http://localhost:{args.port}")
    print(f"\n  POST /tools        Register tool schemas")
    print(f"  POST /query        Run inference (tools cached)")
    print(f"  GET  /status       Cache state")
    print(f"  DELETE /tools      Clear cache")
    print(f"  GET  /sample-tools Demo tools")
    print(f"  GET  /             Web UI")
    print()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
