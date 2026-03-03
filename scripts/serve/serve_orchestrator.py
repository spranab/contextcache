#!/usr/bin/env python3
"""FastAPI server for the Confidence-Gated Route-First Orchestrator.

Two products, one server:

  1. ROUTE-ONLY (POST /route)
     Just tool detection + confidence. You handle the rest.
     - Register tools once, cached as KV state
     - Each query: ~500ms on CPU, returns tool name + confidence
     - Optionally returns the matched tool's schema (include_schema=true)
     - No LLM dependency, no external calls
     Use case: you have your own LLM pipeline and just need fast tool routing.

  2. FULL PIPELINE (POST /query)
     Route -> param extraction -> execute -> LLM synthesis.
     - Local model routes to the right tool (~500ms)
     - External LLM extracts params from a minimal prompt (no full schema)
     - You provide a tool executor (mock, HTTP, or callable)
     - External LLM synthesizes a user-friendly response
     Use case: end-to-end orchestration with any OpenAI-compatible LLM.

Endpoints:
  POST /domains/{domain_id}/tools  Register a tool catalog for a domain
  POST /route                      Route-only (no LLM, ~500ms)
  POST /query                      Full pipeline (route + LLM + execute)
  GET  /health                     Health check
  GET  /domains                    List registered domains
  DELETE /domains/{domain_id}      Remove a domain

Admin:
  POST /admin/llm-config/{id}      Configure server-side LLM credentials
  GET  /admin/llm-config           List LLM configs (keys masked)
  DELETE /admin/llm-config/{id}    Remove LLM config

Usage:
  python scripts/serve/serve_orchestrator.py --config configs/orchestrator_config.yaml
  python scripts/serve/serve_orchestrator.py --config configs/orchestrator_config.yaml --auth
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RegisterDomainRequest(BaseModel):
    tools: list[dict] = Field(
        ..., min_length=1, description="Tool schemas in OpenAI function-calling format"
    )
    system_prompt: str | None = Field(
        None, description="Custom system prompt for this domain's routing"
    )


class RegisterDomainResponse(BaseModel):
    status: str
    domain_id: str
    num_tools: int
    prefix_tokens: int
    state_size_mb: float
    prefill_ms: float


class QueryRequest(BaseModel):
    domain_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=4096)
    tool_executor: str = Field(
        "mock", description="'mock', HTTP URL, or 'callable'"
    )
    llm_provider: str | None = Field(
        None, description="'claude' or 'openai'. Uses server config if not set."
    )
    llm_api_key: str | None = Field(
        None, description="API key. Uses server config if not set."
    )
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_headers: dict[str, str] | None = None
    llm_system_prompt: str | None = None
    enable_thinking: bool | None = Field(
        None,
        description=(
            "Enable LLM thinking/reasoning mode for deeper analysis. "
            "Slower but can improve quality for complex queries. "
            "Overrides server config when set. Default: server config."
        ),
    )


class RouteOnlyRequest(BaseModel):
    domain_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(5, ge=1, le=20)
    include_schema: bool = Field(
        False, description="Include the matched tool's schema in the response"
    )


class ToolCallDetail(BaseModel):
    tool_name: str
    arguments: dict
    result: dict | str | None = None
    confidence: float
    confidence_level: str
    route_ms: float
    execute_ms: float
    llm_ms: float


class QueryResponse(BaseModel):
    final_response: str
    tool_calls: list[ToolCallDetail]
    confidence: float
    confidence_level: str
    num_tool_calls: int
    timings: dict
    thinking: str | None = Field(
        None, description="LLM reasoning trace (when enable_thinking=true)"
    )


class RouteOnlyResponse(BaseModel):
    tool_name: str
    confidence: float
    top_candidates: list[dict]
    tool_schema: dict | None = Field(
        None, description="Schema of the matched tool (when include_schema=true)"
    )
    timings: dict


class DomainInfo(BaseModel):
    domain_id: str
    num_tools: int
    tool_names: list[str]
    state_size_mb: float
    prefix_tokens: int
    query_count: int
    registered_at: float


class LLMConfigRequest(BaseModel):
    provider: str = Field(..., description="'claude' or 'openai'")
    api_key: str = Field(..., min_length=1)
    model: str | None = None
    base_url: str | None = None
    extra_headers: dict[str, str] | None = None
    system_prompt: str | None = None


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------


class OrchestratorServerState:
    def __init__(self):
        self.router = None
        self.orchestrator = None
        self.llm_config_store = None
        self._loading: bool = False
        self._config: dict = {}


state = OrchestratorServerState()
_startup_args: dict = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = _startup_args.get("config_path")
    if not config_path:
        raise RuntimeError("--config is required")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    state._config = cfg

    # Load LLM config
    from context_cache.llm_config import LLMConfigStore

    llm_config_path = cfg.get("llm_config_path")
    if llm_config_path and Path(llm_config_path).exists():
        state.llm_config_store = LLMConfigStore.from_json(llm_config_path)
        print(f"  LLM config: loaded from {llm_config_path}")
    else:
        state.llm_config_store = LLMConfigStore.from_env()
        if state.llm_config_store._default:
            print("  LLM config: loaded from environment (CONTEXTCACHE_LLM_*)")
        else:
            print(
                "  LLM config: none (pass credentials per-request or use llm_config_path)"
            )

    # Create router
    from context_cache.tool_router import ToolRouter

    router_cfg = cfg.get("router", {})
    state.router = ToolRouter(
        model_path=router_cfg["model_path"],
        n_ctx=router_cfg.get("n_ctx", 16384),
        n_threads=router_cfg.get("n_threads", 4),
        n_batch=router_cfg.get("n_batch", 2048),
        system_prompt=router_cfg.get("system_prompt"),
        max_domains=router_cfg.get("max_domains", 20),
    )

    # Load model
    state._loading = True
    print("Loading router model...")
    load_ms = await state.router.load_model()
    state._loading = False
    print(f"Router model loaded in {load_ms:.0f}ms")

    # Pre-register domains from config
    for domain in cfg.get("preload_domains", []):
        tools_path = domain["tools_path"]
        with open(tools_path, encoding="utf-8") as f:
            tools = json.load(f)
        print(f"Pre-registering domain '{domain['domain_id']}' ({len(tools)} tools)...")
        result = await state.router.register_tools(
            domain["domain_id"],
            tools,
            domain.get("system_prompt"),
        )
        print(
            f"  Domain '{domain['domain_id']}' ready: "
            f"{result['num_tools']} tools, {result['prefill_ms']:.0f}ms prefill, "
            f"{result['state_size_mb']:.1f}MB state"
        )

    # Create orchestrator
    from context_cache.orchestrator import Orchestrator, OrchestratorConfig

    orch_cfg = OrchestratorConfig.from_dict(cfg.get("orchestrator", {}))
    state.orchestrator = Orchestrator(
        router=state.router,
        llm_config_store=state.llm_config_store,
        config=orch_cfg,
    )
    print("Orchestrator ready.")

    yield

    # Cleanup
    if state.router:
        await state.router.close()
    print("Shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(
    title="ContextCache Orchestrator",
    version="0.1.0",
    description=(
        "Confidence-gated route-first tool orchestrator. "
        "Routes queries to tools via a local model (~550ms), "
        "then uses an external LLM for param extraction and synthesis."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_metrics_collector = None
_key_store = None


def _setup_auth():
    """Install auth middleware if --auth flag is set."""
    global _metrics_collector, _key_store
    from context_cache.middleware import (
        APIKeyStore,
        AuthRateLimitMiddleware,
        MetricsCollector,
        SlidingWindowRateLimiter,
    )

    _metrics_collector = MetricsCollector()
    rate_limiter = SlidingWindowRateLimiter(default_rpm=60)

    keys_path = ROOT / "api_keys.json"
    env_key = os.environ.get("CONTEXTCACHE_API_KEY")

    if keys_path.exists():
        _key_store = APIKeyStore.from_json(keys_path)
        print(f"  Auth: loaded {_key_store.num_keys} API key(s) from {keys_path}")
    elif env_key:
        _key_store = APIKeyStore.from_env_key(env_key)
        print("  Auth: using CONTEXTCACHE_API_KEY from environment")
    else:
        import secrets

        temp_key = secrets.token_urlsafe(32)
        _key_store = APIKeyStore()
        _key_store.add_key(temp_key, name="dev-key", rate_limit=120, role="admin")
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


STATIC_DIR = ROOT / "scripts" / "serve" / "static"


@app.get("/")
def ui_redirect():
    """Redirect root to the admin UI."""
    return FileResponse(STATIC_DIR / "orchestrator.html")


@app.get("/ui")
def ui_page():
    """Serve the orchestrator admin UI."""
    return FileResponse(STATIC_DIR / "orchestrator.html")


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": (
            "healthy"
            if state.router and state.router._loaded
            else ("loading" if state._loading else "not_started")
        ),
        "router_loaded": state.router._loaded if state.router else False,
        "num_domains": len(state.router._domains) if state.router else 0,
    }


@app.post("/domains/{domain_id}/tools", response_model=RegisterDomainResponse)
async def register_domain(domain_id: str, req: RegisterDomainRequest):
    """Register a tool catalog for a domain.

    This triggers cold prefill of all tool schemas (~44-106s for 50 tools).
    The result is a cached KV state that makes subsequent queries fast (~550ms).
    """
    if not state.router or not state.router._loaded:
        raise HTTPException(503, "Router model not loaded yet")

    result = await state.router.register_tools(
        domain_id, req.tools, req.system_prompt
    )
    return RegisterDomainResponse(
        status="registered",
        domain_id=domain_id,
        num_tools=result["num_tools"],
        prefix_tokens=result["prefix_tokens"],
        state_size_mb=result["state_size_mb"],
        prefill_ms=result["prefill_ms"],
    )


@app.post("/route", response_model=RouteOnlyResponse)
async def route_only(req: RouteOnlyRequest):
    """Route a query to a tool without executing it or calling an LLM.

    This is the lightweight product: just tool detection + confidence.
    The caller handles param extraction, execution, and synthesis.

    Use include_schema=true to get the matched tool's schema back, so you
    can pass it to your own LLM for param extraction.

    Returns the tool name, confidence score, top candidates, and optionally
    the matched tool's schema (~500ms on CPU).
    """
    if not state.router or not state.router._loaded:
        raise HTTPException(503, "Router model not loaded yet")
    if req.domain_id not in state.router._domains:
        raise HTTPException(404, f"Domain '{req.domain_id}' not registered")

    result = await state.router.route(req.domain_id, req.query, top_k=req.top_k)

    # Optionally include the matched tool's schema
    tool_schema = None
    if req.include_schema:
        domain = state.router._domains[req.domain_id]
        for t in domain.tools:
            func = t.get("function", t)
            if func.get("name") == result.tool_name:
                tool_schema = t
                break

    return RouteOnlyResponse(
        tool_name=result.tool_name,
        confidence=result.confidence,
        top_candidates=[
            {"name": name, "probability": round(prob, 4)}
            for name, prob in result.top_candidates
        ],
        tool_schema=tool_schema,
        timings={
            "route_ms": result.route_ms,
            "restore_ms": result.restore_ms,
            "prefill_ms": result.prefill_ms,
            "gen_ms": result.gen_ms,
        },
    )


@app.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    """Full pipeline: route -> param extraction -> execute -> LLM synthesis.

    The local model routes to the right tool (~500ms, cached KV state).
    The external LLM extracts params from a minimal prompt (just param
    names/types — not the full schema), then synthesizes the final response.
    The LLM never sees the full tool catalog.
    """
    if not state.orchestrator:
        raise HTTPException(503, "Orchestrator not ready")
    if req.domain_id not in state.router._domains:
        raise HTTPException(404, f"Domain '{req.domain_id}' not registered")

    try:
        result = await state.orchestrator.process_query(
            domain_id=req.domain_id,
            query=req.query,
            tool_executor=req.tool_executor,
            llm_api_key=req.llm_api_key,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
            llm_base_url=req.llm_base_url,
            llm_extra_headers=req.llm_headers,
            llm_system_prompt=req.llm_system_prompt,
            enable_thinking=req.enable_thinking,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return QueryResponse(
        final_response=result.final_response,
        tool_calls=[
            ToolCallDetail(
                tool_name=tc.tool_name,
                arguments=tc.arguments,
                result=tc.result,
                confidence=tc.route_confidence,
                confidence_level=tc.confidence_level.value,
                route_ms=tc.route_ms,
                execute_ms=tc.execute_ms,
                llm_ms=tc.llm_ms,
            )
            for tc in result.tool_calls
        ],
        confidence=result.confidence,
        confidence_level=result.confidence_level,
        num_tool_calls=result.num_tool_calls,
        timings=result.timings,
        thinking=result.thinking,
    )


@app.get("/domains")
async def list_domains():
    """List all registered domains."""
    if not state.router:
        return {"num_domains": 0, "domains": []}
    domains = await state.router.list_domains()
    return {"num_domains": len(domains), "domains": domains}


@app.delete("/domains/{domain_id}")
async def remove_domain(domain_id: str):
    """Remove a domain and its cached state."""
    if not state.router:
        raise HTTPException(503, "Router not ready")
    removed = await state.router.unregister_domain(domain_id)
    if not removed:
        raise HTTPException(404, f"Domain '{domain_id}' not found")
    return {"status": "removed", "domain_id": domain_id}


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------


@app.get("/admin/metrics")
def get_metrics():
    """Request metrics per API key."""
    if _metrics_collector is None:
        return {
            "auth_enabled": False,
            "message": "Auth not enabled. Start with --auth flag.",
        }
    return _metrics_collector.get_metrics()


@app.post("/admin/llm-config/{domain_id}")
async def set_llm_config(domain_id: str, req: LLMConfigRequest):
    """Configure server-side LLM credentials for a domain."""
    if not state.llm_config_store:
        raise HTTPException(500, "LLM config store not initialized")

    from context_cache.llm_config import LLMConfig

    config = LLMConfig(
        provider=req.provider,
        api_key=req.api_key,
        model=req.model,
        base_url=req.base_url,
        extra_headers=req.extra_headers or {},
        system_prompt=req.system_prompt,
    )
    state.llm_config_store.set(domain_id, config)
    return {"status": "configured", "domain_id": domain_id}


@app.get("/admin/llm-config")
def list_llm_configs():
    """List all LLM configs (API keys masked)."""
    if not state.llm_config_store:
        return {"configs": []}
    return {"configs": state.llm_config_store.list_configured()}


@app.delete("/admin/llm-config/{domain_id}")
def remove_llm_config(domain_id: str):
    """Remove LLM config for a domain."""
    if not state.llm_config_store:
        raise HTTPException(500, "LLM config store not initialized")
    state.llm_config_store.remove(domain_id)
    return {"status": "removed", "domain_id": domain_id}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ContextCache Orchestrator Server")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to orchestrator config YAML",
    )
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--auth", action="store_true", help="Enable API key auth")
    args = parser.parse_args()

    # Load config for host/port defaults
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _startup_args["config_path"] = args.config

    if args.auth:
        _setup_auth()

    host = args.host or cfg.get("host", "0.0.0.0")
    port = args.port or cfg.get("port", 8422)

    print(f"\nStarting ContextCache Orchestrator on {host}:{port}")
    print(f"Config: {args.config}")
    print(f"Auth: {'enabled' if args.auth else 'disabled'}")
    print()

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
