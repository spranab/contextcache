#!/usr/bin/env python3
"""Demo: FastAPI app that uses ContextCache Orchestrator as a backend.

Shows how to build your own API on top of the orchestrator — a common
pattern where your app handles business logic and the orchestrator
handles tool routing + LLM orchestration.

Architecture:
  User -> Your FastAPI App -> ContextCache Orchestrator -> Tools + LLM
                                    |
                             Local 2B model routes (~500ms)
                             External LLM synthesizes (~1-2s)

Prerequisites:
  pip install fastapi uvicorn httpx
  # Start the orchestrator first:
  python scripts/serve/serve_orchestrator.py --config configs/orchestrator_config.yaml

Usage:
  python examples/fastapi_integration.py
  # Then open http://localhost:8000/docs for the Swagger UI
"""

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="My Retail API", version="1.0.0")

ORCHESTRATOR_URL = "http://localhost:8422"
DOMAIN_ID = "retail"


# ── Your app's request/response models ───────────────────────


class ChatRequest(BaseModel):
    message: str
    enable_thinking: bool = False


class ChatResponse(BaseModel):
    reply: str
    tool_used: str | None = None
    tool_args: dict | None = None
    confidence: float
    latency_ms: float


class ToolLookupRequest(BaseModel):
    message: str


class ToolLookupResponse(BaseModel):
    tool_name: str
    confidence: float
    parameters: dict | None = None
    latency_ms: float


# ── Endpoints ────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat endpoint: user message in, assistant response out.

    Behind the scenes, the orchestrator routes to the right tool,
    extracts params, executes, and synthesizes a response.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{ORCHESTRATOR_URL}/query",
            json={
                "domain_id": DOMAIN_ID,
                "query": req.message,
                "tool_executor": "mock",
                # Using Ollama as the LLM — swap these for Claude/OpenAI
                "llm_provider": "openai",
                "llm_api_key": "ollama",
                "llm_model": "qwen3.5:4b",
                "llm_base_url": "http://localhost:11434/v1",
                "enable_thinking": req.enable_thinking,
            },
        )

    if resp.status_code == 404:
        raise HTTPException(
            503,
            f"Domain '{DOMAIN_ID}' not registered. "
            "Register tools first via POST /setup",
        )
    resp.raise_for_status()
    data = resp.json()

    tool_call = data["tool_calls"][0] if data["tool_calls"] else None
    return ChatResponse(
        reply=data["final_response"],
        tool_used=tool_call["tool_name"] if tool_call else None,
        tool_args=tool_call["arguments"] if tool_call else None,
        confidence=data["confidence"],
        latency_ms=data["timings"].get("total_ms", 0),
    )


@app.post("/lookup", response_model=ToolLookupResponse)
async def lookup_tool(req: ToolLookupRequest):
    """Route-only: find the right tool without calling any LLM.

    Use this when you want to handle param extraction yourself,
    or just need to know which tool matches a user intent.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{ORCHESTRATOR_URL}/route",
            json={
                "domain_id": DOMAIN_ID,
                "query": req.message,
                "include_schema": True,
            },
        )

    if resp.status_code == 404:
        raise HTTPException(503, f"Domain '{DOMAIN_ID}' not registered")
    resp.raise_for_status()
    data = resp.json()

    # Extract parameter names from the matched tool's schema
    params = None
    if data.get("tool_schema"):
        props = (
            data["tool_schema"]
            .get("function", {})
            .get("parameters", {})
            .get("properties", {})
        )
        params = {
            name: info.get("description", info.get("type", ""))
            for name, info in props.items()
        }

    return ToolLookupResponse(
        tool_name=data["tool_name"],
        confidence=data["confidence"],
        parameters=params,
        latency_ms=data["timings"]["route_ms"],
    )


@app.post("/setup")
async def setup():
    """Register the retail tool catalog with the orchestrator.

    Call this once at startup. Subsequent calls are idempotent
    (re-registers with the same tools).
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "check_inventory",
                "description": "Check product inventory levels by SKU or product name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "string", "description": "Product name or SKU"},
                    },
                    "required": ["product"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_order_status",
                "description": "Get the current status of a customer order by order ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "Order ID"},
                    },
                    "required": ["order_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "Search products by keyword, category, or price range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keywords"},
                        "max_price": {"type": "number", "description": "Max price filter"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_discount",
                "description": "Apply a percentage discount to an order",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "Order ID"},
                        "discount_percent": {"type": "number", "description": "Discount %"},
                    },
                    "required": ["order_id"],
                },
            },
        },
    ]

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{ORCHESTRATOR_URL}/domains/{DOMAIN_ID}/tools",
            json={"tools": tools},
        )
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    import uvicorn
    print("Starting retail API on http://localhost:8000")
    print("  POST /chat    — full pipeline (route + LLM)")
    print("  POST /lookup  — route-only (no LLM)")
    print("  POST /setup   — register tools (call once)")
    print("\nSwagger UI: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
