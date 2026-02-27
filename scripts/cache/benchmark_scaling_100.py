#!/usr/bin/env python3
"""Extended scaling benchmark: 5 to 100 tools.

Measures cache-hit vs full-prefill TTFT at multiple tool counts to show
that cached TTFT stays flat while full prefill grows quadratically.

Outputs JSON results + generates matplotlib charts.

Usage:
    python scripts/cache/benchmark_scaling_100.py --config configs/context_cache_config.yaml
    python scripts/cache/benchmark_scaling_100.py --config configs/context_cache_config.yaml --tool-counts 5 10 20 30 50 75 100
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from context_cache.cache_config import ContextCacheConfig
from context_cache.context_cache import ContextCacheModel


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def collect_unique_tools(data_dir: Path) -> list[str]:
    """Collect unique tool schemas from all gisting test files."""
    seen = set()
    tools = []
    for fname in ["test_seen_gisting.jsonl", "test_held_out_gisting.jsonl", "test_unseen_gisting.jsonl"]:
        path = data_dir / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                for s in ex.get("tool_schemas", []):
                    if s not in seen:
                        seen.add(s)
                        tools.append(s)
    return tools


def collect_queries(data_dir: Path, max_queries: int = 100) -> list[str]:
    """Collect tool-call queries from test data."""
    queries = []
    seen = set()
    for fname in ["test_seen_gisting.jsonl", "test_held_out_gisting.jsonl", "test_unseen_gisting.jsonl"]:
        path = data_dir / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                gold = ex.get("assistant_response", "")
                is_tc = "<functioncall>" in gold or "<tool_call>" in gold or (
                    gold.strip().startswith("{") and '"name"' in gold
                )
                if is_tc:
                    q = ex.get("user_query", "")
                    if q and q not in seen:
                        seen.add(q)
                        queries.append(q)
                        if len(queries) >= max_queries:
                            return queries
    return queries


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def _percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = (len(s) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def benchmark_one_size(
    model: ContextCacheModel,
    tool_schemas: list[str],
    queries: list[str],
    max_new_tokens: int = 256,
    warmup: int = 2,
) -> dict:
    """Benchmark a single tool count. Returns timing dict."""
    n_tools = len(tool_schemas)
    n_queries = len(queries)
    print(f"\n{'='*60}")
    print(f"  {n_tools} tools, {n_queries} queries")
    print(f"{'='*60}")

    # Clear previous cache
    model._group_cache.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # --- Phase 1: Compile (first call, cache miss) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, timings_compile = model.generate_group_cached(
        context_texts=tool_schemas,
        user_query=queries[0],
        max_new_tokens=max_new_tokens,
    )
    torch.cuda.synchronize()
    compile_total_ms = (time.perf_counter() - t0) * 1000

    # Measure cache size
    cache_mem_bytes = 0
    prefix_tokens = 0
    for gk, (cached_layers, plen) in model._group_cache.items():
        prefix_tokens = plen
        for k, v in cached_layers:
            cache_mem_bytes += k.nelement() * k.element_size()
            cache_mem_bytes += v.nelement() * v.element_size()
    cache_mb = cache_mem_bytes / (1024 * 1024)

    print(f"  Compile: {compile_total_ms:.0f} ms total, cache={cache_mb:.0f} MB, {prefix_tokens} prefix tokens")

    # --- Phase 2: Warmup ---
    for i in range(min(warmup, n_queries)):
        model.generate_group_cached(
            context_texts=tool_schemas,
            user_query=queries[i % n_queries],
            max_new_tokens=max_new_tokens,
        )

    # --- Phase 3: Cache-hit TTFT ---
    hit_link = []
    hit_prefill = []
    print(f"  Cache-hit pass ({n_queries} queries)...")
    for i, q in enumerate(queries):
        torch.cuda.synchronize()
        _, timings = model.generate_group_cached(
            context_texts=tool_schemas,
            user_query=q,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.synchronize()
        assert timings.get("cache_hit", False), f"Expected cache hit on query {i}"
        hit_link.append(timings.get("link_ms", 0))
        hit_prefill.append(timings.get("prefill_query_ms", 0))
        if (i + 1) % 10 == 0:
            avg = _mean(hit_link) + _mean(hit_prefill)
            print(f"    [{i+1}/{n_queries}] avg TTFT: {avg:.0f} ms")

    # --- Phase 4: Full-prefill TTFT ---
    fp_prefill = []
    fp_tokens = []
    print(f"  Full-prefill pass ({n_queries} queries)...")
    for i, q in enumerate(queries):
        torch.cuda.synchronize()
        _, timings = model.generate_full_prefill(
            context_texts=tool_schemas,
            user_query=q,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.synchronize()
        fp_prefill.append(timings.get("prefill_ms", 0))
        if "prompt_tokens" in timings:
            fp_tokens.append(timings["prompt_tokens"])
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{n_queries}] avg TTFT: {_mean(fp_prefill):.0f} ms")

    # --- Compile results ---
    cached_ttft = _mean(hit_link) + _mean(hit_prefill)
    full_ttft = _mean(fp_prefill)
    speedup = full_ttft / max(cached_ttft, 0.001)
    savings = full_ttft - cached_ttft

    result = {
        "num_tools": n_tools,
        "num_queries": n_queries,
        "prefix_tokens": prefix_tokens,
        "cache_size_mb": round(cache_mb, 1),
        "compile_ms": round(compile_total_ms, 1),
        "cached_ttft": {
            "mean": round(cached_ttft, 1),
            "std": round(_std([l + p for l, p in zip(hit_link, hit_prefill)]), 1),
            "p50": round(_percentile(hit_link, 50) + _percentile(hit_prefill, 50), 1),
            "p95": round(_percentile(hit_link, 95) + _percentile(hit_prefill, 95), 1),
            "link_mean": round(_mean(hit_link), 2),
            "prefill_mean": round(_mean(hit_prefill), 1),
        },
        "full_prefill_ttft": {
            "mean": round(full_ttft, 1),
            "std": round(_std(fp_prefill), 1),
            "p50": round(_percentile(fp_prefill, 50), 1),
            "p95": round(_percentile(fp_prefill, 95), 1),
            "prompt_tokens_mean": round(_mean(fp_tokens), 0) if fp_tokens else None,
        },
        "speedup": round(speedup, 2),
        "savings_per_request_ms": round(savings, 1),
        "break_even_requests": round(compile_total_ms / max(savings, 0.001), 2),
    }

    print(f"\n  RESULT: cached={cached_ttft:.0f}ms  full={full_ttft:.0f}ms  "
          f"speedup={speedup:.1f}x  savings={savings:.0f}ms/req")
    return result


def main():
    parser = argparse.ArgumentParser(description="Extended scaling benchmark (5-100 tools)")
    parser.add_argument("--config", type=str, default="configs/context_cache_config.yaml")
    parser.add_argument("--tool-counts", type=int, nargs="+",
                        default=[5, 10, 20, 30, 50, 75, 100])
    parser.add_argument("--queries-per-size", type=int, default=15,
                        help="Queries per tool count (default 15)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=str,
                        default="eval_results/context_cache_benchmark")
    args = parser.parse_args()

    # Load config and bump max positions for large tool sets
    config = ContextCacheConfig.from_yaml(args.config)
    max_tools = max(args.tool_counts)
    if max_tools > 30:
        # Qwen3-8B supports 32K natively (rope_theta=1M)
        config.model.max_seq_length = 32768
        config.rope.max_position = 32768
        print(f"Extended max_seq_length to 32768 for {max_tools}-tool benchmark")

    print("Loading model...")
    model = ContextCacheModel(config)

    # Collect tools and queries
    data_dir = Path("data/processed")
    all_tools = collect_unique_tools(data_dir)
    queries = collect_queries(data_dir, max_queries=args.queries_per_size)
    print(f"Available: {len(all_tools)} unique tools, {len(queries)} queries")

    if len(all_tools) < max_tools:
        print(f"WARNING: Only {len(all_tools)} tools available, max requested is {max_tools}")

    # Run benchmarks
    results = []
    for n in args.tool_counts:
        if n > len(all_tools):
            print(f"\nSKIPPING {n} tools (only {len(all_tools)} available)")
            continue
        tool_subset = all_tools[:n]
        result = benchmark_one_size(
            model, tool_subset, queries,
            max_new_tokens=args.max_new_tokens,
            warmup=2,
        )
        results.append(result)

        # Save after each size (in case of crash)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "benchmark_scaling_100.json", "w") as f:
            json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print("SCALING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Tools':>6} {'Tokens':>7} {'Cache MB':>9} {'Compile':>9} "
          f"{'Cached':>8} {'Full PF':>8} {'Speedup':>8} {'Save/req':>9} {'Break-even':>11}")
    print("-" * 80)
    for r in results:
        print(f"{r['num_tools']:>6} {r['prefix_tokens']:>7} {r['cache_size_mb']:>8.0f} "
              f"{r['compile_ms']:>8.0f}ms "
              f"{r['cached_ttft']['mean']:>7.0f}ms {r['full_prefill_ttft']['mean']:>7.0f}ms "
              f"{r['speedup']:>7.1f}x {r['savings_per_request_ms']:>8.0f}ms "
              f"{r['break_even_requests']:>10.1f}")

    print(f"\nResults saved to {output_dir / 'benchmark_scaling_100.json'}")


if __name__ == "__main__":
    main()
