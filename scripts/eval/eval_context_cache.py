#!/usr/bin/env python3
"""Evaluate ContextCache vs baselines on tool-calling benchmarks.

Conditions:
  A. ContextCache: NoPE KV composition with deferred RoPE
  B. Full Prefill: all schemas as text, standard forward

Metrics:
  - Tool Selection Accuracy (TSA)
  - Parameter F1
  - Exact Match
  - False Positive / False Negative rates
  - TTFT (time-to-first-token) and total latency

Usage:
    python scripts/eval/eval_context_cache.py --config configs/context_cache_config.yaml
    python scripts/eval/eval_context_cache.py --config configs/context_cache_config.yaml --condition cached --max-examples 100
    python scripts/eval/eval_context_cache.py --config configs/context_cache_config.yaml --tool-calls-only
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

# Reuse tool call parsing from existing eval
from scripts.eval.eval_gisting import compute_param_f1, extract_tool_call


@torch.no_grad()
def evaluate_split(
    model: ContextCacheModel,
    data_path: Path,
    split_name: str,
    condition: str = "cached",
    max_examples: int = 200,
    output_dir: Path | None = None,
    tool_calls_only: bool = False,
    fp_sample: int = 30,
    max_new_tokens: int = 256,
) -> dict:
    """Evaluate on a test split.

    Args:
        model: ContextCacheModel instance.
        data_path: Path to test JSONL file.
        split_name: Name of the split (for logging).
        condition: "cached" (ContextCache) or "full_prefill" (baseline).
        max_examples: Maximum tool-call examples to evaluate.
        output_dir: Where to save per-example results.
        tool_calls_only: If True, prioritize tool-call examples.
        fp_sample: Number of non-tool examples for false positive testing.
        max_new_tokens: Max tokens to generate per example.

    Returns:
        Metrics dict.
    """
    print(f"\n--- Evaluating {split_name} [{condition}] ---")

    # Load dataset
    print(f"  Loading {data_path}...")
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} examples")

    # Select evaluation indices
    if tool_calls_only:
        tc_indices = []
        non_tc_indices = []
        for i, ex in enumerate(examples):
            if ex.get("target_tool") and ex.get("is_tool_call", False):
                tc_indices.append(i)
            else:
                non_tc_indices.append(i)
        tc_indices = tc_indices[:max_examples]
        non_tc_indices = non_tc_indices[:fp_sample]
        eval_indices = tc_indices + non_tc_indices
        print(f"  Tool-calls-only mode: {len(tc_indices)} TC + {len(non_tc_indices)} non-TC")
    else:
        eval_indices = list(range(min(max_examples, len(examples))))

    n = len(eval_indices)
    print(f"  Evaluating {n} examples...")

    # Pre-compile all tool schemas in the dataset
    # IMPORTANT: compile using compact JSON (matching apply_chat_template output)
    if condition == "cached":
        print("  Pre-compiling tool schemas...")
        # Get the constant prefix (system prompt + tool instructions header)
        # so tools are compiled with the real context they'll see at link time
        tool_prefix = model.get_tool_prefix()
        compiled_schemas = set()
        for idx in eval_indices:
            ex = examples[idx]
            for schema_text in ex.get("tool_schemas", []):
                try:
                    schema = json.loads(schema_text)
                    name = schema.get("function", {}).get("name", "unknown")
                    # Use compact JSON to match chat template format
                    compact = json.dumps(schema, ensure_ascii=False)
                except json.JSONDecodeError:
                    name = "unknown"
                    compact = schema_text
                # Include trailing \n to match full prompt format
                # (each tool schema is on its own line in the <tools> block)
                compact_nl = compact + "\n"
                if compact_nl not in compiled_schemas:
                    model.compile_context(compact_nl, name, "tool_schema",
                                          prefix_text=tool_prefix)
                    compiled_schemas.add(compact_nl)
        print(f"  Compiled {len(compiled_schemas)} unique tool schemas")

        # Preload to GPU
        if model.config.cache.preload_to_gpu:
            model.kv_store.preload_to_gpu(device=str(model.device))
            print(f"  GPU cache: {model.kv_store.gpu_cache_size_mb():.1f} MB")

    # Evaluate
    results = []
    correct_tool = 0
    num_tool_calls = 0
    false_positives = 0
    false_negatives = 0
    num_non_tool = 0
    param_f1_sum = 0.0
    param_f1_count = 0
    exact_matches = 0
    total_link_ms = 0.0
    total_prefill_ms = 0.0
    total_decode_ms = 0.0

    for progress_idx, idx in enumerate(eval_indices):
        ex = examples[idx]
        tool_schemas_raw = ex.get("tool_schemas", [])
        user_query = ex.get("user_query", "")
        is_tool_call = ex.get("is_tool_call", False)
        target_tool = ex.get("target_tool")
        assistant_response = ex.get("assistant_response", "")

        # Normalize schemas to compact JSON (matching compilation format)
        tool_schemas = []
        for s in tool_schemas_raw:
            try:
                tool_schemas.append(json.dumps(json.loads(s), ensure_ascii=False))
            except json.JSONDecodeError:
                tool_schemas.append(s)

        # Generate
        try:
            if condition == "cached":
                response, timings = model.generate(
                    context_texts=tool_schemas,
                    user_query=user_query,
                    max_new_tokens=max_new_tokens,
                )
                total_link_ms += timings.get("link_ms", 0)
                total_prefill_ms += timings.get("prefill_query_ms", 0)
            elif condition == "group_cached":
                response, timings = model.generate_group_cached(
                    context_texts=tool_schemas,
                    user_query=user_query,
                    max_new_tokens=max_new_tokens,
                )
                total_link_ms += timings.get("link_ms", 0)
                total_prefill_ms += timings.get("prefill_query_ms", 0)
            else:
                response, timings = model.generate_full_prefill(
                    context_texts=tool_schemas,
                    user_query=user_query,
                    max_new_tokens=max_new_tokens,
                )
                total_prefill_ms += timings.get("prefill_ms", 0)

            total_decode_ms += timings.get("decode_ms", 0)
        except Exception as e:
            print(f"  Error on example {idx}: {e}")
            response = ""
            timings = {}

        # Parse predictions
        pred_call = extract_tool_call(response)
        pred_name = pred_call["name"] if pred_call else None
        pred_args = pred_call.get("arguments", {}) if pred_call else {}

        # Parse gold
        gold_call = extract_tool_call(assistant_response) if is_tool_call else None
        gold_name = target_tool or (gold_call["name"] if gold_call else None)
        gold_args = gold_call.get("arguments", {}) if gold_call else {}

        # Score
        if is_tool_call:
            num_tool_calls += 1
            if pred_name == gold_name:
                correct_tool += 1
                # Compute param F1
                _, _, f1 = compute_param_f1(pred_args, gold_args)
                param_f1_sum += f1
                param_f1_count += 1
                if f1 == 1.0 and pred_name == gold_name:
                    exact_matches += 1
            elif pred_name is None:
                false_negatives += 1
        else:
            num_non_tool += 1
            if pred_name is not None:
                false_positives += 1

        result = {
            "idx": idx,
            "is_tool_call": is_tool_call,
            "target_tool": target_tool,
            "predicted": response[:500],
            "pred_name": pred_name,
            "gold_name": gold_name,
            "correct": (pred_name == gold_name) if is_tool_call else (pred_name is None),
            "timings": timings,
        }
        results.append(result)

        if (progress_idx + 1) % 10 == 0 or progress_idx == n - 1:
            tsa = correct_tool / max(num_tool_calls, 1)
            print(
                f"  [{progress_idx+1}/{n}] TSA: {tsa:.3f} ({correct_tool}/{num_tool_calls}), "
                f"FP: {false_positives}, FN: {false_negatives}"
            )

    # Compute final metrics
    metrics = {
        "split": split_name,
        "condition": condition,
        "num_examples": n,
        "num_tool_calls": num_tool_calls,
        "num_non_tool": num_non_tool,
        "tool_selection_accuracy": correct_tool / max(num_tool_calls, 1),
        "parameter_f1": param_f1_sum / max(param_f1_count, 1),
        "exact_match": exact_matches / max(num_tool_calls, 1),
        "false_positive_rate": false_positives / max(num_non_tool, 1),
        "false_negative_rate": false_negatives / max(num_tool_calls, 1),
        "correct_tool": correct_tool,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "avg_link_ms": total_link_ms / max(n, 1),
        "avg_prefill_ms": total_prefill_ms / max(n, 1),
        "avg_decode_ms": total_decode_ms / max(n, 1),
    }

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"results_{split_name}_{condition}.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        with open(output_dir / f"summary_{split_name}_{condition}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ContextCache")
    parser.add_argument("--config", type=str, default="configs/context_cache_config.yaml")
    parser.add_argument("--output-dir", type=str, default="eval_results/context_cache")
    parser.add_argument("--condition", type=str, nargs="+", default=["cached", "full_prefill"],
                        choices=["cached", "group_cached", "full_prefill"])
    parser.add_argument("--splits", type=str, nargs="+", default=None)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--tool-calls-only", action="store_true")
    parser.add_argument("--fp-sample", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    config = ContextCacheConfig.from_yaml(args.config)
    model = ContextCacheModel(config)
    output_dir = Path(args.output_dir)

    # Determine splits
    all_splits = {
        "test_seen": config.eval.test_splits[0] if len(config.eval.test_splits) > 0 else None,
        "test_held_out": config.eval.test_splits[1] if len(config.eval.test_splits) > 1 else None,
        "test_unseen": config.eval.test_splits[2] if len(config.eval.test_splits) > 2 else None,
    }

    if args.splits:
        splits = {k: v for k, v in all_splits.items() if k in args.splits and v}
    else:
        splits = {k: v for k, v in all_splits.items() if v}

    # Run evaluation
    all_metrics = {}
    for condition in args.condition:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        for split_name, data_path in splits.items():
            if not Path(data_path).exists():
                print(f"  Skipping {split_name}: {data_path} not found")
                continue

            metrics = evaluate_split(
                model=model,
                data_path=Path(data_path),
                split_name=split_name,
                condition=condition,
                max_examples=args.max_examples,
                output_dir=output_dir,
                tool_calls_only=args.tool_calls_only,
                fp_sample=args.fp_sample,
                max_new_tokens=args.max_new_tokens,
            )
            all_metrics[f"{split_name}_{condition}"] = metrics

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Split':<20} {'Condition':<15} {'TSA':>6} {'Param F1':>9} {'EM':>6} {'FPR':>6} {'FNR':>6} {'Link ms':>8} {'Prefill ms':>11}")
    print("-" * 100)
    for key, m in all_metrics.items():
        print(
            f"{m['split']:<20} {m['condition']:<15} "
            f"{m['tool_selection_accuracy']:>6.3f} "
            f"{m['parameter_f1']:>9.3f} "
            f"{m['exact_match']:>6.3f} "
            f"{m['false_positive_rate']:>6.3f} "
            f"{m['false_negative_rate']:>6.3f} "
            f"{m['avg_link_ms']:>8.0f} "
            f"{m['avg_prefill_ms']:>11.0f}"
        )

    # Save overall summary
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary_all.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
