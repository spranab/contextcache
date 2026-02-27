#!/usr/bin/env python3
"""Test client for ContextCache server.

Usage:
  # Start the server first:
  python scripts/serve/serve_context_cache.py

  # Then run this:
  python scripts/serve/test_client.py
  python scripts/serve/test_client.py --base-url http://localhost:8421
"""

import argparse
import json
import time

import requests


def main():
    parser = argparse.ArgumentParser(description="Test ContextCache server")
    parser.add_argument("--base-url", type=str, default="http://localhost:8421")
    args = parser.parse_args()
    base = args.base_url

    print(f"Testing ContextCache server at {base}\n")

    # 1. Health check
    print("=" * 60)
    print("1. Health check")
    r = requests.get(f"{base}/health")
    print(f"   Status: {r.status_code}")
    print(f"   {r.json()}")

    # 2. Register tools
    print("\n" + "=" * 60)
    print("2. Register tools (POST /tools)")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_age",
                "description": "Calculate a person's age from their birthdate",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "birthdate": {"type": "string", "description": "Birth date in YYYY-MM-DD format"},
                    },
                    "required": ["birthdate"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "num_results": {"type": "integer", "description": "Number of results"},
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    t0 = time.perf_counter()
    r = requests.post(f"{base}/tools", json={"tools": tools})
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"   Status: {r.status_code} ({elapsed:.0f}ms round-trip)")
    resp = r.json()
    print(f"   {json.dumps(resp, indent=2)}")

    # 3. Status check
    print("\n" + "=" * 60)
    print("3. Status (GET /status)")
    r = requests.get(f"{base}/status")
    print(f"   Status: {r.status_code}")
    resp = r.json()
    print(f"   Tools loaded: {resp['tools_loaded']}")
    print(f"   Num tools: {resp['num_tools']}")
    print(f"   Tool names: {resp['tool_names']}")
    print(f"   Cache hash: {resp['cache_hash']}")
    print(f"   Cache size: {resp['cache_size_mb']} MB")

    # 4. Run queries (first should be cache hit if tools were already compiled)
    queries = [
        "What is the weather in New York City?",
        "How old is someone born on 1990-05-15?",
        "Search for the latest news about AI",
        "What is the capital of France?",  # No tool needed
    ]

    print("\n" + "=" * 60)
    print("4. Run queries (POST /query)")
    for i, query in enumerate(queries):
        print(f"\n   --- Query {i+1}: {query}")
        t0 = time.perf_counter()
        r = requests.post(f"{base}/query", json={"query": query, "max_new_tokens": 128})
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            resp = r.json()
            print(f"   Cache hit: {resp['cache_hit']}")
            print(f"   Response: {resp['response'][:200]}")
            print(f"   Timings: link={resp['timings'].get('link_ms', 0):.0f}ms, "
                  f"prefill={resp['timings'].get('prefill_query_ms', 0):.0f}ms, "
                  f"decode={resp['timings'].get('decode_ms', 0):.0f}ms")
            print(f"   Total round-trip: {elapsed:.0f}ms")
        else:
            print(f"   ERROR {r.status_code}: {r.text}")

    # 5. Re-register same tools (should be already_cached)
    print("\n" + "=" * 60)
    print("5. Re-register same tools (should be already_cached)")
    t0 = time.perf_counter()
    r = requests.post(f"{base}/tools", json={"tools": tools})
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"   Status: {r.status_code} ({elapsed:.0f}ms)")
    print(f"   {r.json()}")

    # 6. Clear tools
    print("\n" + "=" * 60)
    print("6. Clear tools (DELETE /tools)")
    r = requests.delete(f"{base}/tools")
    print(f"   Status: {r.status_code}")
    print(f"   {r.json()}")

    # 7. Query without tools (should fail)
    print("\n" + "=" * 60)
    print("7. Query without tools (should fail with 400)")
    r = requests.post(f"{base}/query", json={"query": "Hello"})
    print(f"   Status: {r.status_code}")
    print(f"   {r.json()}")

    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    main()
