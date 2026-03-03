#!/usr/bin/env python3
"""Demo: Retail assistant powered by ContextCache Orchestrator.

A complete working example showing how to:
  1. Register a tool catalog with the orchestrator
  2. Route user queries to the right tool (~500ms)
  3. Run the full pipeline with LLM param extraction + synthesis (~3s)
  4. Handle confidence levels (high, low, no-tool)

Prerequisites:
  - Orchestrator server running:
      python scripts/serve/serve_orchestrator.py --config configs/orchestrator_config.yaml
  - For full pipeline: Ollama running with a model pulled:
      ollama pull qwen3.5:4b

Usage:
  # Interactive mode
  python examples/retail_assistant.py

  # Route-only mode (no LLM needed)
  python examples/retail_assistant.py --route-only

  # Custom LLM endpoint
  python examples/retail_assistant.py --llm-model qwen3.5:9b

  # Use Claude or OpenAI instead of Ollama
  python examples/retail_assistant.py --llm-provider claude --llm-api-key sk-ant-...
"""

import argparse
import json
import sys

import requests

# ── Configuration ────────────────────────────────────────────

ORCHESTRATOR_URL = "http://localhost:8422"
DOMAIN_ID = "retail"

# Tool catalog — standard OpenAI function-calling format
RETAIL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check product inventory levels by SKU or product name",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Product name or SKU",
                    },
                    "warehouse": {
                        "type": "string",
                        "description": "Warehouse location (optional)",
                    },
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
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a percentage discount or coupon code to an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order to apply discount to",
                    },
                    "discount_percent": {
                        "type": "number",
                        "description": "Discount percentage (e.g., 10 for 10%)",
                    },
                    "coupon_code": {
                        "type": "string",
                        "description": "Coupon code (optional)",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products by keyword, category, or price range",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or keyword",
                    },
                    "category": {
                        "type": "string",
                        "description": "Product category filter",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price filter",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_return",
            "description": "Process a product return or exchange for a customer",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Original order ID",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for return",
                    },
                    "action": {
                        "type": "string",
                        "description": "Action: 'refund' or 'exchange'",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
]


# ── Orchestrator Client ──────────────────────────────────────


class OrchestratorClient:
    """Minimal client for the ContextCache Orchestrator REST API."""

    def __init__(self, base_url: str = ORCHESTRATOR_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> dict:
        r = self.session.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def register_tools(
        self, domain_id: str, tools: list[dict], system_prompt: str | None = None
    ) -> dict:
        """Register a tool catalog for a domain (triggers cold prefill)."""
        body = {"tools": tools}
        if system_prompt:
            body["system_prompt"] = system_prompt
        r = self.session.post(
            f"{self.base_url}/domains/{domain_id}/tools",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def list_domains(self) -> list[dict]:
        r = self.session.get(f"{self.base_url}/domains", timeout=10)
        r.raise_for_status()
        return r.json().get("domains", [])

    def route(
        self, domain_id: str, query: str, top_k: int = 5, include_schema: bool = False
    ) -> dict:
        """Route-only: tool detection + confidence, no LLM needed."""
        r = self.session.post(
            f"{self.base_url}/route",
            json={
                "domain_id": domain_id,
                "query": query,
                "top_k": top_k,
                "include_schema": include_schema,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def query(
        self,
        domain_id: str,
        query: str,
        llm_provider: str = "openai",
        llm_api_key: str = "ollama",
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        tool_executor: str = "mock",
        enable_thinking: bool = False,
    ) -> dict:
        """Full pipeline: route -> extract params -> execute -> synthesize."""
        body = {
            "domain_id": domain_id,
            "query": query,
            "tool_executor": tool_executor,
            "llm_provider": llm_provider,
            "llm_api_key": llm_api_key,
        }
        if llm_model:
            body["llm_model"] = llm_model
        if llm_base_url:
            body["llm_base_url"] = llm_base_url
        if enable_thinking:
            body["enable_thinking"] = True
        r = self.session.post(
            f"{self.base_url}/query", json=body, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()


# ── Demo Application ─────────────────────────────────────────


def register_tools(client: OrchestratorClient, domain_id: str) -> bool:
    """Register the retail tool catalog, or skip if already registered."""
    # Check if domain already exists
    domains = client.list_domains()
    for d in domains:
        if d["domain_id"] == domain_id:
            print(f"  Domain '{domain_id}' already registered "
                  f"({d['num_tools']} tools, {d['query_count']} queries)")
            return True

    print(f"  Registering {len(RETAIL_TOOLS)} tools under domain '{domain_id}'...")
    print("  (This triggers cold prefill — one-time cost, may take a few seconds)")
    result = client.register_tools(domain_id, RETAIL_TOOLS)
    print(f"  Registered: {result['num_tools']} tools, "
          f"{result['prefix_tokens']} prefix tokens, "
          f"{result['state_size_mb']:.1f}MB state, "
          f"{result['prefill_ms']:.0f}ms prefill")
    return True


def demo_route_only(client: OrchestratorClient, domain_id: str):
    """Demo: route queries to tools without any LLM."""
    print("\n" + "=" * 60)
    print("  Route-Only Demo (no LLM needed)")
    print("=" * 60)

    queries = [
        "Do we have iPhone 16 Pro in stock?",
        "What's the status of order ORD-2024-1234?",
        "Search for wireless headphones under $100",
        "I want to return order ORD-5555 because it arrived damaged",
        "What's the weather like today?",  # Should get low confidence
    ]

    for q in queries:
        result = client.route(domain_id, q, include_schema=True)
        conf = result["confidence"]
        level = "HIGH" if conf >= 0.7 else ("LOW" if conf >= 0.2 else "NO_TOOL")

        print(f"\n  Q: {q}")
        print(f"  -> {result['tool_name']} (confidence: {conf:.3f}, {level})")
        print(f"     Route time: {result['timings']['route_ms']:.0f}ms")

        if result.get("top_candidates"):
            top3 = result["top_candidates"][:3]
            cands = ", ".join(f"{c['name']}({c['probability']:.2f})" for c in top3)
            print(f"     Candidates: {cands}")


def demo_full_pipeline(
    client: OrchestratorClient,
    domain_id: str,
    llm_provider: str,
    llm_api_key: str,
    llm_model: str | None,
    llm_base_url: str | None,
):
    """Demo: full pipeline with LLM param extraction + synthesis."""
    print("\n" + "=" * 60)
    print("  Full Pipeline Demo (with LLM)")
    print(f"  LLM: {llm_model or 'default'}")
    print("=" * 60)

    queries = [
        "Do we have iPhone 16 Pro in stock?",
        "Apply a 15% discount to order ORD-5678",
        "Search for wireless headphones under $100",
        "What's the weather like today?",  # No-tool query
    ]

    for q in queries:
        print(f"\n  Q: {q}")
        try:
            result = client.query(
                domain_id=domain_id,
                query=q,
                llm_provider=llm_provider,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
            )

            print(f"  Confidence: {result['confidence']:.3f} ({result['confidence_level']})")

            if result["tool_calls"]:
                tc = result["tool_calls"][0]
                print(f"  Tool: {tc['tool_name']}")
                print(f"  Args: {json.dumps(tc['arguments'])}")

            # Truncate long responses
            resp = result["final_response"]
            if len(resp) > 200:
                resp = resp[:200] + "..."
            print(f"  Response: {resp}")

            timings = result["timings"]
            print(f"  Timings: route={timings.get('route_ms', 0):.0f}ms, "
                  f"params={timings.get('param_extraction_ms', 0):.0f}ms, "
                  f"synth={timings.get('synthesis_ms', 0):.0f}ms, "
                  f"total={timings.get('total_ms', 0):.0f}ms")

        except requests.exceptions.HTTPError as e:
            print(f"  Error: {e.response.status_code} - {e.response.text[:200]}")
        except Exception as e:
            print(f"  Error: {e}")


def interactive_mode(
    client: OrchestratorClient,
    domain_id: str,
    route_only: bool,
    llm_provider: str,
    llm_api_key: str,
    llm_model: str | None,
    llm_base_url: str | None,
):
    """Interactive REPL: type queries, see results."""
    mode = "route-only" if route_only else "full pipeline"
    print(f"\n  Interactive mode ({mode}). Type 'quit' to exit.\n")

    while True:
        try:
            query = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        try:
            if route_only:
                result = client.route(domain_id, query, include_schema=True)
                conf = result["confidence"]
                print(f"  -> Tool: {result['tool_name']}")
                print(f"     Confidence: {conf:.3f}")
                print(f"     Time: {result['timings']['route_ms']:.0f}ms")
                if result.get("tool_schema"):
                    params = (
                        result["tool_schema"]
                        .get("function", {})
                        .get("parameters", {})
                        .get("properties", {})
                    )
                    if params:
                        print(f"     Params: {', '.join(params.keys())}")
            else:
                result = client.query(
                    domain_id=domain_id,
                    query=query,
                    llm_provider=llm_provider,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model,
                    llm_base_url=llm_base_url,
                )
                print(f"  Assistant: {result['final_response']}")
                if result["tool_calls"]:
                    tc = result["tool_calls"][0]
                    print(f"  [tool={tc['tool_name']}, "
                          f"conf={result['confidence']:.2f}, "
                          f"{result['timings'].get('total_ms', 0):.0f}ms]")
        except Exception as e:
            print(f"  Error: {e}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Retail assistant demo using ContextCache Orchestrator"
    )
    parser.add_argument(
        "--url", default=ORCHESTRATOR_URL,
        help="Orchestrator server URL (default: http://localhost:8422)",
    )
    parser.add_argument(
        "--domain", default=DOMAIN_ID,
        help="Domain ID for tool registration (default: retail)",
    )
    parser.add_argument(
        "--route-only", action="store_true",
        help="Route-only mode (no LLM needed, just tool detection)",
    )
    parser.add_argument(
        "--llm-provider", default="openai",
        help="LLM provider: 'openai' (for Ollama/OpenAI-compatible) or 'claude'",
    )
    parser.add_argument(
        "--llm-api-key", default="ollama",
        help="LLM API key (default: 'ollama' for local Ollama)",
    )
    parser.add_argument(
        "--llm-model", default="qwen3.5:4b",
        help="LLM model name (default: qwen3.5:4b via Ollama)",
    )
    parser.add_argument(
        "--llm-base-url", default="http://localhost:11434/v1",
        help="LLM base URL (default: Ollama at http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo queries then exit (no interactive mode)",
    )
    args = parser.parse_args()

    client = OrchestratorClient(args.url)

    # Step 1: Check server health
    print("\n  Connecting to orchestrator...")
    try:
        health = client.health()
        if health["status"] != "healthy":
            print(f"  Server not ready: {health['status']}")
            sys.exit(1)
        print(f"  Connected. {health['num_domains']} domain(s) registered.")
    except requests.exceptions.ConnectionError:
        print(f"  Cannot connect to {args.url}")
        print("  Start the server with:")
        print("    python scripts/serve/serve_orchestrator.py "
              "--config configs/orchestrator_config.yaml")
        sys.exit(1)

    # Step 2: Register tools
    register_tools(client, args.domain)

    # Step 3: Run demo or interactive mode
    if args.demo:
        if args.route_only:
            demo_route_only(client, args.domain)
        else:
            demo_route_only(client, args.domain)
            demo_full_pipeline(
                client, args.domain,
                args.llm_provider, args.llm_api_key,
                args.llm_model, args.llm_base_url,
            )
    else:
        interactive_mode(
            client, args.domain, args.route_only,
            args.llm_provider, args.llm_api_key,
            args.llm_model, args.llm_base_url,
        )

    print("\n  Done!")


if __name__ == "__main__":
    main()
