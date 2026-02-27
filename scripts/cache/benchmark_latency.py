#!/usr/bin/env python3
"""Benchmark ContextCache latency for deployment scenario.

Simulates the real deployment pattern:
  1. Tools deployed once (one-time compile cost)
  2. Many user requests with the same tool set (all cache hits)

Measures:
  - Compile time (one-time, per tool set)
  - Per-request TTFT: cache-hit link + suffix prefill
  - Per-request TTFT: full prefill (baseline)
  - Storage cost (GPU memory for cached KV)
  - Amortization: break-even point and daily savings

Usage:
    python scripts/cache/benchmark_latency.py --config configs/context_cache_config.yaml
    python scripts/cache/benchmark_latency.py --config configs/context_cache_config.yaml --num-queries 100 --tool-counts 10 20 50 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from context_cache.cache_config import ContextCacheConfig
from context_cache.context_cache import ContextCacheModel


def load_test_examples(data_path: Path, max_examples: int = 200) -> list[dict]:
    """Load test examples, deriving is_tool_call from gold response."""
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            gold = ex.get("assistant_response", "")
            ex["is_tool_call"] = (
                "<functioncall>" in gold or "<tool_call>" in gold
                or (gold.strip().startswith("{") and '"name"' in gold)
            )
            examples.append(ex)
            if len(examples) >= max_examples:
                break
    return examples


def extract_tool_set(examples: list[dict], max_tools: int = 50) -> list[str]:
    """Extract a representative tool set: union of all tool schemas from examples.

    Args:
        examples: Test examples with tool_schemas field.
        max_tools: Maximum number of tools to include. With eager attention,
            large tool sets (>50) create O(n^2) attention bottlenecks.
    """
    seen = set()
    tool_schemas = []
    for ex in examples:
        for schema_text in ex.get("tool_schemas", []):
            try:
                schema = json.loads(schema_text)
                compact = json.dumps(schema, ensure_ascii=False)
            except json.JSONDecodeError:
                compact = schema_text
            if compact not in seen:
                seen.add(compact)
                tool_schemas.append(compact)
                if len(tool_schemas) >= max_tools:
                    return tool_schemas
    return tool_schemas


def extract_queries(examples: list[dict], max_queries: int = 100) -> list[dict]:
    """Extract user queries (tool-call examples only)."""
    queries = []
    for ex in examples:
        if ex.get("is_tool_call"):
            queries.append({
                "user_query": ex["user_query"],
                "target_tool": ex.get("target_tool"),
                "assistant_response": ex.get("assistant_response", ""),
            })
            if len(queries) >= max_queries:
                break
    return queries


def benchmark_tool_set(
    model: ContextCacheModel,
    tool_schemas: list[str],
    queries: list[dict],
    max_new_tokens: int = 256,
    warmup: int = 2,
) -> dict:
    """Benchmark a single tool set configuration.

    Returns dict with compile_ms, cache_hit timings, full_prefill timings, storage info.
    """
    results = {
        "num_tools": len(tool_schemas),
        "num_queries": len(queries),
    }

    # Clear any previous group cache
    model._group_cache.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ==========================================
    # Phase 1: Measure compile time (first call)
    # ==========================================
    first_query = queries[0]["user_query"]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    response_compile, timings_compile = model.generate_group_cached(
        context_texts=tool_schemas,
        user_query=first_query,
        max_new_tokens=max_new_tokens,
    )
    torch.cuda.synchronize()
    compile_total_ms = (time.perf_counter() - t0) * 1000

    results["compile_ms"] = timings_compile.get("link_ms", 0)
    results["compile_cache_hit"] = timings_compile.get("cache_hit", False)
    results["compile_total_ms"] = compile_total_ms

    # Measure GPU memory used by the group cache
    cache_mem_bytes = 0
    for group_key, (cached_layers, prefix_len) in model._group_cache.items():
        for k, v in cached_layers:
            cache_mem_bytes += k.nelement() * k.element_size()
            cache_mem_bytes += v.nelement() * v.element_size()
    results["cache_size_mb"] = cache_mem_bytes / (1024 * 1024)
    results["cache_prefix_tokens"] = list(model._group_cache.values())[0][1] if model._group_cache else 0

    print(f"  Compile: {results['compile_ms']:.0f} ms (link), cache_hit={results['compile_cache_hit']}")
    print(f"  Cache: {results['cache_size_mb']:.1f} MB, {results['cache_prefix_tokens']} prefix tokens")

    # ==========================================
    # Phase 2: Warmup cache-hit path
    # ==========================================
    for i in range(min(warmup, len(queries))):
        model.generate_group_cached(
            context_texts=tool_schemas,
            user_query=queries[i % len(queries)]["user_query"],
            max_new_tokens=max_new_tokens,
        )

    # ==========================================
    # Phase 3: Benchmark cache-hit requests
    # ==========================================
    cache_hit_link_ms = []
    cache_hit_prefill_ms = []
    cache_hit_decode_ms = []
    cache_hit_total_ms = []

    print(f"  Running {len(queries)} cache-hit queries...")
    for i, q in enumerate(queries):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        response, timings = model.generate_group_cached(
            context_texts=tool_schemas,
            user_query=q["user_query"],
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

        assert timings.get("cache_hit", False), f"Expected cache hit on query {i}"

        cache_hit_link_ms.append(timings.get("link_ms", 0))
        cache_hit_prefill_ms.append(timings.get("prefill_query_ms", 0))
        cache_hit_decode_ms.append(timings.get("decode_ms", 0))
        cache_hit_total_ms.append(total_ms)

        if (i + 1) % 20 == 0:
            avg_ttft = sum(cache_hit_link_ms) / len(cache_hit_link_ms) + \
                       sum(cache_hit_prefill_ms) / len(cache_hit_prefill_ms)
            print(f"    [{i+1}/{len(queries)}] avg TTFT: {avg_ttft:.0f} ms")

    results["cache_hit"] = {
        "link_ms": {"mean": _mean(cache_hit_link_ms), "std": _std(cache_hit_link_ms),
                    "min": min(cache_hit_link_ms), "max": max(cache_hit_link_ms),
                    "p50": _percentile(cache_hit_link_ms, 50), "p95": _percentile(cache_hit_link_ms, 95)},
        "prefill_ms": {"mean": _mean(cache_hit_prefill_ms), "std": _std(cache_hit_prefill_ms),
                       "min": min(cache_hit_prefill_ms), "max": max(cache_hit_prefill_ms),
                       "p50": _percentile(cache_hit_prefill_ms, 50), "p95": _percentile(cache_hit_prefill_ms, 95)},
        "decode_ms": {"mean": _mean(cache_hit_decode_ms), "std": _std(cache_hit_decode_ms)},
        "total_ms": {"mean": _mean(cache_hit_total_ms), "std": _std(cache_hit_total_ms)},
        "ttft_ms": {"mean": _mean(cache_hit_link_ms) + _mean(cache_hit_prefill_ms),
                    "p50": _percentile(cache_hit_link_ms, 50) + _percentile(cache_hit_prefill_ms, 50),
                    "p95": _percentile(cache_hit_link_ms, 95) + _percentile(cache_hit_prefill_ms, 95)},
    }

    print(f"  Cache-hit TTFT: {results['cache_hit']['ttft_ms']['mean']:.0f} ms "
          f"(link={results['cache_hit']['link_ms']['mean']:.0f} + "
          f"prefill={results['cache_hit']['prefill_ms']['mean']:.0f})")

    # ==========================================
    # Phase 4: Benchmark full prefill baseline
    # ==========================================
    full_prefill_prefill_ms = []
    full_prefill_decode_ms = []
    full_prefill_total_ms = []
    prompt_tokens_list = []

    print(f"  Running {len(queries)} full-prefill queries...")
    for i, q in enumerate(queries):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        response, timings = model.generate_full_prefill(
            context_texts=tool_schemas,
            user_query=q["user_query"],
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

        full_prefill_prefill_ms.append(timings.get("prefill_ms", 0))
        full_prefill_decode_ms.append(timings.get("decode_ms", 0))
        full_prefill_total_ms.append(total_ms)
        if "prompt_tokens" in timings:
            prompt_tokens_list.append(timings["prompt_tokens"])

        if (i + 1) % 20 == 0:
            avg_ttft = sum(full_prefill_prefill_ms) / len(full_prefill_prefill_ms)
            print(f"    [{i+1}/{len(queries)}] avg TTFT: {avg_ttft:.0f} ms")

    results["full_prefill"] = {
        "prefill_ms": {"mean": _mean(full_prefill_prefill_ms), "std": _std(full_prefill_prefill_ms),
                       "min": min(full_prefill_prefill_ms), "max": max(full_prefill_prefill_ms),
                       "p50": _percentile(full_prefill_prefill_ms, 50), "p95": _percentile(full_prefill_prefill_ms, 95)},
        "decode_ms": {"mean": _mean(full_prefill_decode_ms), "std": _std(full_prefill_decode_ms)},
        "total_ms": {"mean": _mean(full_prefill_total_ms), "std": _std(full_prefill_total_ms)},
        "ttft_ms": {"mean": _mean(full_prefill_prefill_ms),
                    "p50": _percentile(full_prefill_prefill_ms, 50),
                    "p95": _percentile(full_prefill_prefill_ms, 95)},
        "prompt_tokens": {"mean": _mean(prompt_tokens_list)} if prompt_tokens_list else {},
    }

    print(f"  Full-prefill TTFT: {results['full_prefill']['ttft_ms']['mean']:.0f} ms")

    # ==========================================
    # Phase 5: Compute amortization
    # ==========================================
    compile_cost = results["compile_ms"]
    savings_per_request = results["full_prefill"]["ttft_ms"]["mean"] - results["cache_hit"]["ttft_ms"]["mean"]
    break_even = compile_cost / max(savings_per_request, 0.001)

    results["amortization"] = {
        "compile_cost_ms": compile_cost,
        "savings_per_request_ms": savings_per_request,
        "break_even_requests": break_even,
        "speedup_factor": results["full_prefill"]["ttft_ms"]["mean"] / max(results["cache_hit"]["ttft_ms"]["mean"], 0.001),
        # Savings for different request volumes (ms)
        "savings_100_requests_ms": 100 * savings_per_request - compile_cost,
        "savings_1000_requests_ms": 1000 * savings_per_request - compile_cost,
        "savings_10000_requests_ms": 10000 * savings_per_request - compile_cost,
    }

    print(f"  Speedup: {results['amortization']['speedup_factor']:.1f}x TTFT")
    print(f"  Break-even: {results['amortization']['break_even_requests']:.1f} requests")
    print(f"  Savings per request: {savings_per_request:.0f} ms")

    return results


def benchmark_scaling(
    model: ContextCacheModel,
    all_tool_schemas: list[str],
    queries: list[dict],
    tool_counts: list[int],
    max_new_tokens: int = 256,
    queries_per_size: int = 20,
) -> list[dict]:
    """Benchmark scaling across different numbers of tools."""
    scaling_results = []

    for n_tools in tool_counts:
        if n_tools > len(all_tool_schemas):
            print(f"  SKIP {n_tools} tools (only {len(all_tool_schemas)} available)")
            continue

        print(f"\n--- {n_tools} tools ---")
        tool_subset = all_tool_schemas[:n_tools]
        query_subset = queries[:queries_per_size]

        result = benchmark_tool_set(
            model, tool_subset, query_subset,
            max_new_tokens=max_new_tokens, warmup=2,
        )
        scaling_results.append(result)

    return scaling_results


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = (len(s) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def main():
    parser = argparse.ArgumentParser(description="Benchmark ContextCache latency")
    parser.add_argument("--config", type=str, default="configs/context_cache_config.yaml")
    parser.add_argument("--split", type=str, default="test_seen",
                        choices=["test_seen", "test_held_out", "test_unseen"])
    parser.add_argument("--num-queries", type=int, default=50,
                        help="Number of queries for the main benchmark")
    parser.add_argument("--max-tools", type=int, default=20,
                        help="Max tools for main benchmark (eager attn is O(n^2))")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tool-counts", type=int, nargs="+", default=None,
                        help="Tool counts for scaling experiment (e.g., 5 10 20 50)")
    parser.add_argument("--queries-per-size", type=int, default=20,
                        help="Queries per tool count in scaling experiment")
    parser.add_argument("--output-dir", type=str, default="eval_results/context_cache_benchmark")
    args = parser.parse_args()

    config = ContextCacheConfig.from_yaml(args.config)
    print("Loading model...")
    model = ContextCacheModel(config)

    # Load test data
    split_map = {
        "test_seen": config.eval.test_splits[0],
        "test_held_out": config.eval.test_splits[1] if len(config.eval.test_splits) > 1 else None,
        "test_unseen": config.eval.test_splits[2] if len(config.eval.test_splits) > 2 else None,
    }
    data_path = split_map.get(args.split)
    if not data_path or not Path(data_path).exists():
        print(f"ERROR: Split {args.split} data not found at {data_path}")
        return

    print(f"Loading test data from {data_path}...")
    examples = load_test_examples(Path(data_path), max_examples=500)
    print(f"  Loaded {len(examples)} examples")

    # Extract tool set and queries
    all_tool_schemas = extract_tool_set(examples, max_tools=args.max_tools)
    queries = extract_queries(examples, max_queries=args.num_queries)
    print(f"  {len(all_tool_schemas)} unique tool schemas, {len(queries)} tool-call queries")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Main benchmark: full tool set
    # ==========================================
    print(f"\n{'='*60}")
    print(f"Main benchmark: {len(all_tool_schemas)} tools, {len(queries)} queries")
    print(f"{'='*60}")

    main_results = benchmark_tool_set(
        model, all_tool_schemas, queries,
        max_new_tokens=args.max_new_tokens, warmup=3,
    )

    with open(output_dir / "benchmark_main.json", "w") as f:
        json.dump(main_results, f, indent=2)
    print(f"\nMain results saved to {output_dir}/benchmark_main.json")

    # ==========================================
    # Scaling benchmark (if requested)
    # ==========================================
    if args.tool_counts:
        print(f"\n{'='*60}")
        print(f"Scaling benchmark: {args.tool_counts} tools")
        print(f"{'='*60}")

        scaling_queries = queries[:args.queries_per_size]
        scaling_results = benchmark_scaling(
            model, all_tool_schemas, scaling_queries,
            tool_counts=args.tool_counts,
            max_new_tokens=args.max_new_tokens,
            queries_per_size=args.queries_per_size,
        )

        with open(output_dir / "benchmark_scaling.json", "w") as f:
            json.dump(scaling_results, f, indent=2)
        print(f"\nScaling results saved to {output_dir}/benchmark_scaling.json")

    # ==========================================
    # Print summary
    # ==========================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Tools: {main_results['num_tools']}")
    print(f"Queries: {main_results['num_queries']}")
    print(f"Cache size: {main_results['cache_size_mb']:.1f} MB ({main_results['cache_prefix_tokens']} tokens)")
    print(f"")
    print(f"Compile (one-time):    {main_results['compile_ms']:.0f} ms")
    print(f"Cache-hit TTFT:        {main_results['cache_hit']['ttft_ms']['mean']:.0f} ms "
          f"(p50={main_results['cache_hit']['ttft_ms']['p50']:.0f}, "
          f"p95={main_results['cache_hit']['ttft_ms']['p95']:.0f})")
    print(f"  Link:                {main_results['cache_hit']['link_ms']['mean']:.0f} ms")
    print(f"  Suffix prefill:      {main_results['cache_hit']['prefill_ms']['mean']:.0f} ms")
    print(f"Full-prefill TTFT:     {main_results['full_prefill']['ttft_ms']['mean']:.0f} ms "
          f"(p50={main_results['full_prefill']['ttft_ms']['p50']:.0f}, "
          f"p95={main_results['full_prefill']['ttft_ms']['p95']:.0f})")
    print(f"")
    print(f"Speedup:               {main_results['amortization']['speedup_factor']:.1f}x")
    print(f"Savings per request:   {main_results['amortization']['savings_per_request_ms']:.0f} ms")
    print(f"Break-even:            {main_results['amortization']['break_even_requests']:.1f} requests")
    print(f"Savings (100 req):     {main_results['amortization']['savings_100_requests_ms']/1000:.1f} sec")
    print(f"Savings (1K req):      {main_results['amortization']['savings_1000_requests_ms']/1000:.1f} sec")
    print(f"Savings (10K req):     {main_results['amortization']['savings_10000_requests_ms']/1000:.1f} sec")


if __name__ == "__main__":
    main()
