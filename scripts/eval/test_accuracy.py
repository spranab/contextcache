#!/usr/bin/env python3
"""Routing accuracy test — register N merchant tools, run known queries, check correctness.

Tests the forced prefix + stop sequence improvements by verifying that the
model consistently selects the correct tool from large tool sets.

Usage:
    # Start the server first:
    python scripts/serve/serve_context_cache.py

    # Run accuracy test (defaults to 100 tools):
    python scripts/eval/test_accuracy.py

    # Custom:
    python scripts/eval/test_accuracy.py --num-tools 50 --base-url http://localhost:8421
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Generate N merchant analytics tools
# ---------------------------------------------------------------------------

MERCHANT_TOOL_TEMPLATES = [
    ("gmv_summary", "Get gross merchandise value summary for a time period", {"time_period": {"type": "string", "enum": ["today", "last_7_days", "last_30_days", "last_90_days", "ytd"]}}),
    ("revenue_breakdown", "Break down revenue by category, product, or channel", {"group_by": {"type": "string", "enum": ["category", "product", "channel"]}, "time_period": {"type": "string"}}),
    ("order_count", "Get total number of orders for a time period", {"time_period": {"type": "string"}, "status": {"type": "string", "enum": ["all", "completed", "pending", "cancelled"]}}),
    ("average_order_value", "Calculate the average order value", {"time_period": {"type": "string"}, "currency": {"type": "string"}}),
    ("top_products", "Get top selling products by revenue or units", {"metric": {"type": "string", "enum": ["revenue", "units"]}, "limit": {"type": "integer"}}),
    ("customer_count", "Get number of unique customers", {"time_period": {"type": "string"}, "type": {"type": "string", "enum": ["all", "new", "returning"]}}),
    ("refund_rate", "Get refund rate and refund amount summary", {"time_period": {"type": "string"}}),
    ("conversion_rate", "Get checkout conversion rate", {"time_period": {"type": "string"}, "funnel_step": {"type": "string"}}),
    ("cart_abandonment", "Get cart abandonment rate and analysis", {"time_period": {"type": "string"}}),
    ("shipping_performance", "Get shipping and delivery performance metrics", {"carrier": {"type": "string"}, "time_period": {"type": "string"}}),
    ("inventory_levels", "Check current inventory levels for products", {"product_id": {"type": "string"}, "warehouse": {"type": "string"}}),
    ("stock_alerts", "Get low stock alerts and reorder recommendations", {"threshold": {"type": "integer"}}),
    ("payment_methods", "Breakdown of payment methods used by customers", {"time_period": {"type": "string"}}),
    ("geographic_sales", "Sales distribution by region or country", {"granularity": {"type": "string", "enum": ["country", "state", "city"]}}),
    ("customer_lifetime_value", "Calculate customer lifetime value", {"segment": {"type": "string"}}),
    ("churn_analysis", "Analyze customer churn rate and factors", {"time_period": {"type": "string"}}),
    ("marketing_roi", "Calculate return on marketing spend by channel", {"channel": {"type": "string"}}),
    ("discount_usage", "Analyze discount and coupon usage", {"code": {"type": "string"}}),
    ("product_reviews", "Get product review summaries and ratings", {"product_id": {"type": "string"}}),
    ("support_tickets", "Get customer support ticket metrics", {"status": {"type": "string", "enum": ["open", "closed", "pending"]}}),
]


def generate_tools(n: int) -> list[dict]:
    """Generate N merchant analytics tools (cycles through templates, adds suffixes)."""
    tools = []
    for i in range(n):
        base_idx = i % len(MERCHANT_TOOL_TEMPLATES)
        name, desc, params = MERCHANT_TOOL_TEMPLATES[base_idx]
        suffix = f"_v{i // len(MERCHANT_TOOL_TEMPLATES) + 1}" if i >= len(MERCHANT_TOOL_TEMPLATES) else ""
        tool = {
            "type": "function",
            "function": {
                "name": f"{name}{suffix}",
                "description": f"{desc}{' (variant ' + str(i // len(MERCHANT_TOOL_TEMPLATES) + 1) + ')' if suffix else ''}",
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": list(params.keys())[:1],
                },
            },
        }
        tools.append(tool)
    return tools


# ---------------------------------------------------------------------------
# Test queries with expected tool selections
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    ("What's my GMV for the last 30 days?", "gmv_summary"),
    ("Show me revenue breakdown by category", "revenue_breakdown"),
    ("How many orders did we get today?", "order_count"),
    ("What's the average order value this month?", "average_order_value"),
    ("What are our top selling products?", "top_products"),
    ("How many new customers did we get?", "customer_count"),
    ("What's our refund rate?", "refund_rate"),
    ("What's our checkout conversion rate?", "conversion_rate"),
    ("How many carts were abandoned?", "cart_abandonment"),
    ("How is shipping performance looking?", "shipping_performance"),
    ("Check inventory for product SKU-001", "inventory_levels"),
    ("Are there any low stock alerts?", "stock_alerts"),
    ("What payment methods are customers using?", "payment_methods"),
    ("Show sales by country", "geographic_sales"),
    ("What's the customer lifetime value?", "customer_lifetime_value"),
    ("Analyze customer churn", "churn_analysis"),
    ("What's the ROI on our marketing?", "marketing_roi"),
    ("How are discount codes performing?", "discount_usage"),
    ("Show product review summaries", "product_reviews"),
    ("How many open support tickets?", "support_tickets"),
]


def run_accuracy_test(base_url: str, num_tools: int, verbose: bool = True):
    """Register tools, run queries, measure accuracy."""
    tool_id = "accuracy_test"

    print(f"\n{'=' * 70}")
    print(f"  Routing Accuracy Test — {num_tools} tools")
    print(f"{'=' * 70}\n")

    # 1. Health check
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        health = r.json()
        print(f"Server: {base_url}")
        print(f"Mode:   {health.get('mode', 'unknown')}")
        print(f"Model:  {health.get('model_loaded', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {base_url}: {e}")
        return

    # 2. Generate and register tools
    print(f"\nGenerating {num_tools} merchant analytics tools...")
    tools = generate_tools(num_tools)
    tool_names = [t["function"]["name"] for t in tools]

    print(f"Registering as '{tool_id}'...")
    t0 = time.perf_counter()
    r = requests.post(
        f"{base_url}/v2/tools",
        json={"tool_id": tool_id, "tools": tools},
        timeout=300,
    )
    compile_ms = (time.perf_counter() - t0) * 1000

    if r.status_code != 200:
        print(f"ERROR: Registration failed: {r.status_code} {r.text}")
        return

    resp = r.json()
    print(f"  Compiled in {compile_ms:.0f}ms (server: {resp.get('compile_ms', 0):.0f}ms)")
    print(f"  Cache hash: {resp.get('cache_hash', 'N/A')[:16]}...")
    print(f"  Cache size: {resp.get('cache_size_mb', 0):.1f} MB")

    # 3. Run test queries
    print(f"\nRunning {len(TEST_QUERIES)} test queries...\n")

    correct = 0
    total = 0
    results = []

    for query, expected_tool in TEST_QUERIES:
        # Only test tools that exist in the generated set
        if expected_tool not in tool_names:
            continue

        total += 1
        t0 = time.perf_counter()
        r = requests.post(
            f"{base_url}/route",
            json={"tool_id": tool_id, "query": query, "max_new_tokens": 256},
            timeout=30,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        if r.status_code != 200:
            status = "ERROR"
            selected = "N/A"
            confidence = 0.0
        else:
            data = r.json()
            selections = data.get("selections", [])
            selected = selections[0]["tool_name"] if selections else "none"
            confidence = data.get("confidence", 0.0)
            timings = data.get("timings", {})

            if selected == expected_tool:
                correct += 1
                status = "PASS"
            else:
                status = "FAIL"

        results.append({
            "query": query,
            "expected": expected_tool,
            "selected": selected,
            "status": status,
            "confidence": confidence,
            "latency_ms": latency_ms,
        })

        icon = "+" if status == "PASS" else "-" if status == "FAIL" else "!"
        if verbose:
            print(f"  [{icon}] {query[:50]:50s} → {selected:30s} (expected: {expected_tool}, conf: {confidence:.1f}, {latency_ms:.0f}ms)")

    # 4. Summary
    accuracy = correct / total * 100 if total > 0 else 0
    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    high_conf = sum(1 for r in results if r["confidence"] >= 1.0) / total * 100 if total > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  Results: {correct}/{total} correct ({accuracy:.0f}% accuracy)")
    print(f"  High confidence (1.0): {high_conf:.0f}%")
    print(f"  Avg latency: {avg_latency:.0f}ms")
    print(f"{'=' * 70}")

    # Show failures
    failures = [r for r in results if r["status"] == "FAIL"]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in failures:
            print(f"    {f['query'][:60]}")
            print(f"      Expected: {f['expected']}, Got: {f['selected']} (conf: {f['confidence']:.1f})")

    # 5. Cleanup
    requests.delete(f"{base_url}/v2/tools/{tool_id}", timeout=10)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "high_confidence_pct": high_conf,
        "avg_latency_ms": avg_latency,
        "num_tools": num_tools,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test routing accuracy with merchant tools")
    parser.add_argument("--base-url", type=str, default="http://localhost:8421")
    parser.add_argument("--num-tools", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_accuracy_test(args.base_url, args.num_tools, verbose=not args.quiet)
    if result:
        sys.exit(0 if result["accuracy"] >= 70 else 1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
