#!/usr/bin/env python3
"""Generate paper figures for ContextCache experiments.

Figures (quality/error analysis):
  1. Quality equivalence: ContextCache vs Full Prefill bar chart (TSA, Param F1, EM)
  2. Latency comparison: TTFT breakdown (link + prefill + decode)
  3. Error analysis: FP/FN rates
  4. Split comparison: Seen vs Held-out vs Unseen
  5. Method comparison: Full Prefill vs Prefix Cache vs Gisting vs ContextCache

Figures (deployment/amortization, from benchmark_latency.py output):
  6. TTFT comparison: cache hit vs full prefill (main deployment figure)
  7. Amortization curve: cumulative time savings vs requests
  8. Scaling: TTFT vs number of tools (if scaling data available)
  9. LaTeX deployment table

Usage:
  python scripts/analysis/cache_plots.py --results-dir eval_results/context_cache
  python scripts/analysis/cache_plots.py --results-dir eval_results/context_cache --benchmark-dir eval_results/context_cache_benchmark
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Paper-quality style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.1,
})

CONDITION_COLORS = {
    "cached": "#90CAF9",          # Light blue — per-tool independent (degraded)
    "group_cached": "#2196F3",    # Blue — group caching (our method)
    "full_prefill": "#FF9800",    # Orange — baseline
    "gisting": "#4CAF50",         # Green — gisting comparison
}
CONDITION_LABELS = {
    "cached": "Per-Tool Cache (NoPE)",
    "group_cached": "ContextCache (Ours)",
    "full_prefill": "Full Prefill",
    "gisting": "Tool Gisting (K=8)",
}
SPLIT_COLORS = {
    "test_seen": "#2196F3",
    "test_held_out": "#FF9800",
    "test_unseen": "#F44336",
}
SPLIT_LABELS = {
    "test_seen": "Seen Tools (80)",
    "test_held_out": "Held-out (20)",
    "test_unseen": "Unseen (20)",
}


def load_summary(results_dir: Path) -> dict:
    """Load summary_all.json from ContextCache eval results."""
    summary_path = results_dir / "summary_all.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    # Try individual summary files
    summaries = {}
    for p in results_dir.glob("summary_*.json"):
        if p.name == "summary_all.json":
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
            key = f"{data.get('split', 'unknown')}_{data.get('condition', 'unknown')}"
            summaries[key] = data
    return summaries


def plot_quality_equivalence(metrics: dict, output_dir: Path):
    """Figure 1: Bar chart — All 3 conditions quality metrics.

    Main claim: group_cached matches full_prefill (lossless), per-tool cached degrades.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    metric_keys = ["tool_selection_accuracy", "parameter_f1", "exact_match"]
    metric_labels = ["Tool Selection\nAccuracy", "Parameter\nF1", "Exact\nMatch"]
    conditions = ["cached", "group_cached", "full_prefill"]

    # Find available conditions for primary split
    primary_split = "test_seen"
    available_conds = [c for c in conditions if f"{primary_split}_{c}" in metrics]
    if not available_conds:
        # Try any split
        for key in metrics:
            for c in conditions:
                if key.endswith(f"_{c}"):
                    primary_split = key[: -len(c) - 1]
                    available_conds = [c for c in conditions if f"{primary_split}_{c}" in metrics]
                    break
            if available_conds:
                break
    if not available_conds:
        print("  [SKIP] No data for quality equivalence plot")
        return

    x = np.arange(len(metric_keys))
    n_conds = len(available_conds)
    width = 0.25
    offsets = np.linspace(-(n_conds - 1) / 2 * width, (n_conds - 1) / 2 * width, n_conds)

    for i, cond in enumerate(available_conds):
        key = f"{primary_split}_{cond}"
        m = metrics[key]
        values = [m.get(mk, 0) for mk in metric_keys]
        bars = ax.bar(
            x + offsets[i], values, width,
            label=CONDITION_LABELS.get(cond, cond),
            color=CONDITION_COLORS.get(cond, f"C{i}"),
            edgecolor="white", linewidth=0.5, alpha=0.85,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Quality: ContextCache vs Full Prefill ({primary_split})")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig1_quality_equivalence.{ext}")
    print(f"  Saved fig1_quality_equivalence.pdf/png")
    plt.close(fig)


def plot_latency_breakdown(metrics: dict, output_dir: Path):
    """Figure 2: Stacked bar — TTFT breakdown (link + prefill vs full prefill).

    Shows where time is spent: link (RoPE rotation), query prefill, decode.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Find available splits
    splits_data = {}
    for key, m in metrics.items():
        split = m.get("split", key.rsplit("_", 1)[0])
        cond = m.get("condition", key.rsplit("_", 1)[-1])
        if split not in splits_data:
            splits_data[split] = {}
        splits_data[split][cond] = m

    primary_split = "test_seen" if "test_seen" in splits_data else (list(splits_data.keys())[0] if splits_data else None)
    if not primary_split or not splits_data.get(primary_split):
        print("  [SKIP] No data for latency breakdown plot")
        return

    data = splits_data[primary_split]
    conditions = []
    link_times = []
    prefill_times = []
    decode_times = []

    if "cached" in data:
        conditions.append("Per-Tool Cache\n(NoPE)")
        link_times.append(data["cached"].get("avg_link_ms", 0))
        prefill_times.append(data["cached"].get("avg_prefill_ms", 0))
        decode_times.append(data["cached"].get("avg_decode_ms", 0))

    if "group_cached" in data:
        conditions.append("ContextCache\n(Ours)")
        link_times.append(data["group_cached"].get("avg_link_ms", 0))
        prefill_times.append(data["group_cached"].get("avg_prefill_ms", 0))
        decode_times.append(data["group_cached"].get("avg_decode_ms", 0))

    if "full_prefill" in data:
        conditions.append("Full Prefill")
        link_times.append(0)
        prefill_times.append(data["full_prefill"].get("avg_prefill_ms", 0))
        decode_times.append(data["full_prefill"].get("avg_decode_ms", 0))

    if not conditions:
        print("  [SKIP] No conditions found for latency plot")
        return

    x = np.arange(len(conditions))
    width = 0.5

    bars1 = ax.bar(x, link_times, width, label="Link (RoPE rotation)", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, prefill_times, width, bottom=link_times, label="Query Prefill", color="#FF9800", alpha=0.85)
    bottoms = [l + p for l, p in zip(link_times, prefill_times)]
    bars3 = ax.bar(x, decode_times, width, bottom=bottoms, label="Decode", color="#9E9E9E", alpha=0.85)

    # Add total labels
    for i, (l, p, d) in enumerate(zip(link_times, prefill_times, decode_times)):
        total = l + p + d
        ttft = l + p
        ax.text(i, total + 50, f"Total: {total:.0f}ms\nTTFT: {ttft:.0f}ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Latency Breakdown ({primary_split})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig2_latency_breakdown.{ext}")
    print(f"  Saved fig2_latency_breakdown.pdf/png")
    plt.close(fig)


def plot_error_analysis(metrics: dict, output_dir: Path):
    """Figure 3: FP/FN rates — all conditions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    metric_keys = ["false_positive_rate", "false_negative_rate"]
    metric_labels = ["False Positive Rate\n(non-tool \u2192 tool call)", "False Negative Rate\n(tool call \u2192 no call)"]
    conditions = ["cached", "group_cached", "full_prefill"]

    primary_split = "test_seen"
    available_conds = [c for c in conditions if f"{primary_split}_{c}" in metrics]
    if not available_conds:
        print("  [SKIP] No data for error analysis plot")
        return

    x = np.arange(len(metric_keys))
    n_conds = len(available_conds)
    width = 0.25
    offsets = np.linspace(-(n_conds - 1) / 2 * width, (n_conds - 1) / 2 * width, n_conds)

    for i, cond in enumerate(available_conds):
        key = f"{primary_split}_{cond}"
        m = metrics[key]
        values = [m.get(mk, 0) for mk in metric_keys]
        bars = ax.bar(
            x + offsets[i], values, width,
            label=CONDITION_LABELS.get(cond, cond),
            color=CONDITION_COLORS.get(cond, f"C{i}"),
            edgecolor="white", linewidth=0.5, alpha=0.85,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1] * 1.2))
    ax.set_ylabel("Rate")
    ax.set_title(f"Error Rates ({primary_split})")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig3_error_analysis.{ext}")
    print(f"  Saved fig3_error_analysis.pdf/png")
    plt.close(fig)


def plot_split_comparison(metrics: dict, output_dir: Path):
    """Figure 4: Grouped bar — TSA across seen/held-out/unseen for all conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    conditions = ["cached", "group_cached", "full_prefill"]
    splits = ["test_seen", "test_held_out", "test_unseen"]
    metric_keys = ["tool_selection_accuracy", "parameter_f1", "exact_match"]
    metric_titles = ["Tool Selection Accuracy", "Parameter F1", "Exact Match"]

    # Check which splits have data
    available_splits = []
    for split in splits:
        for cond in conditions:
            if f"{split}_{cond}" in metrics:
                if split not in available_splits:
                    available_splits.append(split)
                break

    if len(available_splits) < 2:
        print("  [SKIP] Need at least 2 splits for split comparison plot")
        return

    available_conds = [c for c in conditions if any(f"{s}_{c}" in metrics for s in available_splits)]
    n_conds = len(available_conds)
    width = 0.25
    offsets = np.linspace(-(n_conds - 1) / 2 * width, (n_conds - 1) / 2 * width, n_conds)
    x = np.arange(len(available_splits))

    for ax_idx, (mk, title) in enumerate(zip(metric_keys, metric_titles)):
        ax = axes[ax_idx]
        for i, cond in enumerate(available_conds):
            values = []
            for split in available_splits:
                key = f"{split}_{cond}"
                values.append(metrics.get(key, {}).get(mk, 0))
            bars = ax.bar(
                x + offsets[i], values, width,
                label=CONDITION_LABELS.get(cond, cond),
                color=CONDITION_COLORS.get(cond, f"C{i}"),
                edgecolor="white", linewidth=0.5, alpha=0.85,
            )
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8, rotation=45,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in available_splits], fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        if ax_idx == 0:
            ax.set_ylabel("Score")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_conds, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig4_split_comparison.{ext}")
    print(f"  Saved fig4_split_comparison.pdf/png")
    plt.close(fig)


def plot_method_comparison(metrics: dict, gisting_results: dict | None, output_dir: Path):
    """Figure 5: Horizontal bar — All methods compared.

    Shows Full Prefill, Per-Tool Cache, Group Cache (ContextCache), and optionally Gisting.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = []
    tsa_values = []
    colors_list = []
    annotations = []

    primary_split = "test_seen"

    # Full prefill
    fp_key = f"{primary_split}_full_prefill"
    if fp_key in metrics:
        methods.append("Full Prefill\n(all schemas as text)")
        tsa_values.append(metrics[fp_key].get("tool_selection_accuracy", 0))
        colors_list.append(CONDITION_COLORS["full_prefill"])
        prefill_ms = metrics[fp_key].get("avg_prefill_ms", 0)
        annotations.append(f"TTFT: {prefill_ms:.0f}ms")

    # Gisting (if available)
    if gisting_results:
        gist_split = gisting_results.get(primary_split, gisting_results.get("test_seen", {}))
        if gist_split:
            methods.append("Tool Gisting (K=8)\n(lossy compression)")
            tsa_values.append(gist_split.get("tool_selection_accuracy", 0))
            colors_list.append(CONDITION_COLORS["gisting"])
            annotations.append("8 tokens/tool")

    # Per-Tool Cache (degraded)
    pt_key = f"{primary_split}_cached"
    if pt_key in metrics:
        methods.append("Per-Tool Cache (NoPE)\n(independent compilation)")
        tsa_values.append(metrics[pt_key].get("tool_selection_accuracy", 0))
        colors_list.append(CONDITION_COLORS["cached"])
        link_ms = metrics[pt_key].get("avg_link_ms", 0)
        prefill_ms = metrics[pt_key].get("avg_prefill_ms", 0)
        annotations.append(f"TTFT: {link_ms + prefill_ms:.0f}ms")

    # ContextCache (group_cached) — our method
    gc_key = f"{primary_split}_group_cached"
    if gc_key in metrics:
        methods.append("ContextCache (Ours)\n(group-cached, lossless)")
        tsa_values.append(metrics[gc_key].get("tool_selection_accuracy", 0))
        colors_list.append(CONDITION_COLORS["group_cached"])
        link_ms = metrics[gc_key].get("avg_link_ms", 0)
        prefill_ms = metrics[gc_key].get("avg_prefill_ms", 0)
        annotations.append(f"TTFT: {link_ms + prefill_ms:.0f}ms")

    if not methods:
        print("  [SKIP] No data for method comparison plot")
        return

    y = np.arange(len(methods))
    bars = ax.barh(y, tsa_values, color=colors_list, alpha=0.85, edgecolor="white", height=0.6)

    for i, (bar, tsa, ann) in enumerate(zip(bars, tsa_values, annotations)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"TSA={tsa:.3f}  |  {ann}", va="center", fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlim(0, 1.35)
    ax.set_xlabel("Tool Selection Accuracy")
    ax.set_title("Method Comparison (test_seen)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig5_method_comparison.{ext}")
    print(f"  Saved fig5_method_comparison.pdf/png")
    plt.close(fig)


def generate_latex_table(metrics: dict, output_dir: Path):
    """Generate LaTeX comparison table for the paper."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{ContextCache evaluation: Quality and latency across conditions and tool splits. "
        r"Group-cached ContextCache matches full prefill quality while enabling persistent caching.}",
        r"\label{tab:context-cache-results}",
        r"\small",
        r"\begin{tabular}{llcccccrrr}",
        r"\toprule",
        r"Split & Condition & TSA$\uparrow$ & PF1$\uparrow$ & EM$\uparrow$ & FPR$\downarrow$ & FNR$\downarrow$ & Link & Prefill & Decode \\",
        r"\midrule",
    ]

    COND_LABELS = {
        "cached": "Per-Tool NoPE",
        "group_cached": r"\textbf{ContextCache}",
        "full_prefill": "Full Prefill",
    }
    conditions = ["cached", "group_cached", "full_prefill"]
    splits = ["test_seen", "test_held_out", "test_unseen"]

    for split in splits:
        for cond in conditions:
            key = f"{split}_{cond}"
            if key not in metrics:
                continue
            m = metrics[key]
            split_label = split.replace("test_", "").replace("_", "-").title()
            cond_label = COND_LABELS.get(cond, cond)
            lines.append(
                f"  {split_label} & {cond_label} "
                f"& {m.get('tool_selection_accuracy', 0):.3f} "
                f"& {m.get('parameter_f1', 0):.3f} "
                f"& {m.get('exact_match', 0):.3f} "
                f"& {m.get('false_positive_rate', 0):.3f} "
                f"& {m.get('false_negative_rate', 0):.3f} "
                f"& {m.get('avg_link_ms', 0):.0f} "
                f"& {m.get('avg_prefill_ms', 0):.0f} "
                f"& {m.get('avg_decode_ms', 0):.0f} \\\\"
            )
        if split != splits[-1]:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    out_path = output_dir / "table_context_cache.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out_path}")


def load_benchmark(benchmark_dir: Path) -> tuple[dict | None, list | None]:
    """Load benchmark results from benchmark_latency.py output."""
    main_path = benchmark_dir / "benchmark_main.json"
    scaling_path = benchmark_dir / "benchmark_scaling.json"
    main_data = None
    scaling_data = None
    if main_path.exists():
        with open(main_path, encoding="utf-8") as f:
            main_data = json.load(f)
    if scaling_path.exists():
        with open(scaling_path, encoding="utf-8") as f:
            scaling_data = json.load(f)
    return main_data, scaling_data


def plot_ttft_comparison(benchmark: dict, output_dir: Path):
    """Figure 6: TTFT comparison — cache hit vs full prefill.

    The main deployment figure. Shows that cache-hit TTFT is a fraction of full prefill.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ch = benchmark["cache_hit"]
    fp = benchmark["full_prefill"]

    # Cache hit: link + prefill
    link_mean = ch["link_ms"]["mean"]
    prefill_mean = ch["prefill_ms"]["mean"]
    cache_ttft = ch["ttft_ms"]["mean"]

    # Full prefill: single prefill
    fp_ttft = fp["ttft_ms"]["mean"]

    labels = ["Full Prefill\n(every request)", "ContextCache\n(cache hit)"]
    x = np.arange(len(labels))
    width = 0.45

    # Stacked bars for cache hit (link + suffix prefill)
    bar_fp = ax.bar(x[0], fp_ttft, width, color=CONDITION_COLORS["full_prefill"],
                    alpha=0.85, edgecolor="white", linewidth=0.5)
    bar_link = ax.bar(x[1], link_mean, width, color="#1565C0",
                      alpha=0.85, edgecolor="white", linewidth=0.5, label="KV Cache Load")
    bar_suffix = ax.bar(x[1], prefill_mean, width, bottom=link_mean, color="#42A5F5",
                        alpha=0.85, edgecolor="white", linewidth=0.5, label="Suffix Prefill")

    # Speedup annotation
    speedup = benchmark["amortization"]["speedup_factor"]
    ax.annotate(
        f"{speedup:.0f}x faster",
        xy=(x[1], cache_ttft), xytext=(x[1] + 0.35, fp_ttft * 0.6),
        fontsize=14, fontweight="bold", color="#1565C0",
        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2),
        ha="left", va="center",
    )

    # Value labels
    ax.text(x[0], fp_ttft + 30, f"{fp_ttft:.0f} ms", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(x[1], cache_ttft + 30, f"{cache_ttft:.0f} ms", ha="center", va="bottom",
            fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Time to First Token (ms)")
    ax.set_title(f"TTFT: ContextCache vs Full Prefill ({benchmark['num_tools']} tools)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, fp_ttft * 1.25)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig6_ttft_comparison.{ext}")
    print(f"  Saved fig6_ttft_comparison.pdf/png")
    plt.close(fig)


def plot_amortization_curve(benchmark: dict, output_dir: Path):
    """Figure 7: Cumulative time savings vs number of requests.

    Shows the deployment economics: compile cost is paid once, savings accumulate.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    compile_cost = benchmark["amortization"]["compile_cost_ms"]
    savings_per_req = benchmark["amortization"]["savings_per_request_ms"]
    break_even = benchmark["amortization"]["break_even_requests"]

    # X axis: number of requests
    max_requests = max(1000, int(break_even * 20))
    requests = np.arange(0, max_requests + 1)

    # Cumulative time: full_prefill vs cache_hit
    fp_ttft = benchmark["full_prefill"]["ttft_ms"]["mean"]
    ch_ttft = benchmark["cache_hit"]["ttft_ms"]["mean"]

    cumulative_fp = requests * fp_ttft  # full prefill: linear from 0
    cumulative_ch = compile_cost + requests * ch_ttft  # cache: compile once + per-request

    # Net savings = cumulative_fp - cumulative_ch
    net_savings = cumulative_fp - cumulative_ch

    # Plot cumulative time
    ax.plot(requests, cumulative_fp / 1000, color=CONDITION_COLORS["full_prefill"],
            linewidth=2, label=f"Full Prefill ({fp_ttft:.0f} ms/req)")
    ax.plot(requests, cumulative_ch / 1000, color=CONDITION_COLORS["group_cached"],
            linewidth=2, label=f"ContextCache ({ch_ttft:.0f} ms/req + {compile_cost:.0f} ms compile)")

    # Mark break-even point
    be_y = (compile_cost + break_even * ch_ttft) / 1000
    ax.axvline(x=break_even, color="#999", linestyle="--", alpha=0.6)
    ax.plot(break_even, be_y, "ko", markersize=8)
    ax.annotate(
        f"Break-even:\n{break_even:.0f} requests",
        xy=(break_even, be_y), xytext=(break_even + max_requests * 0.08, be_y * 1.2),
        fontsize=10, ha="left",
        arrowprops=dict(arrowstyle="->", color="#666"),
    )

    # Shade savings region
    ax.fill_between(requests, cumulative_fp / 1000, cumulative_ch / 1000,
                    where=net_savings > 0, alpha=0.15, color=CONDITION_COLORS["group_cached"])

    # Add savings annotation at right edge
    savings_at_max = net_savings[-1]
    ax.annotate(
        f"Savings at {max_requests} req:\n{savings_at_max/1000:.1f} sec",
        xy=(max_requests, cumulative_ch[-1] / 1000),
        xytext=(max_requests * 0.7, (cumulative_fp[-1] + cumulative_ch[-1]) / 2 / 1000),
        fontsize=10, fontweight="bold", color=CONDITION_COLORS["group_cached"],
        arrowprops=dict(arrowstyle="->", color=CONDITION_COLORS["group_cached"]),
    )

    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Cumulative TTFT (seconds)")
    ax.set_title(f"Amortization: Compile Once, Serve Many ({benchmark['num_tools']} tools)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_requests)
    ax.set_ylim(0)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig7_amortization.{ext}")
    print(f"  Saved fig7_amortization.pdf/png")
    plt.close(fig)


def plot_scaling_ttft(scaling_data: list[dict], output_dir: Path):
    """Figure 8: TTFT vs number of tools — scaling behavior.

    Cache-hit TTFT should stay roughly constant (suffix-only),
    while full-prefill TTFT grows linearly with tool count.
    """
    if not scaling_data or len(scaling_data) < 2:
        print("  [SKIP] Need at least 2 scaling points for scaling plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    tool_counts = [d["num_tools"] for d in scaling_data]
    cache_ttfts = [d["cache_hit"]["ttft_ms"]["mean"] for d in scaling_data]
    fp_ttfts = [d["full_prefill"]["ttft_ms"]["mean"] for d in scaling_data]
    compile_times = [d["compile_ms"] for d in scaling_data]
    cache_sizes = [d["cache_size_mb"] for d in scaling_data]
    speedups = [d["amortization"]["speedup_factor"] for d in scaling_data]

    # Panel 1: TTFT comparison
    ax1.plot(tool_counts, fp_ttfts, "o-", color=CONDITION_COLORS["full_prefill"],
             linewidth=2, markersize=8, label="Full Prefill TTFT")
    ax1.plot(tool_counts, cache_ttfts, "s-", color=CONDITION_COLORS["group_cached"],
             linewidth=2, markersize=8, label="Cache-Hit TTFT")
    ax1.plot(tool_counts, compile_times, "^--", color="#999", linewidth=1.5,
             markersize=7, label="Compile Time (one-time)", alpha=0.7)

    # Speedup annotations
    for i, (tc, sp) in enumerate(zip(tool_counts, speedups)):
        ax1.annotate(f"{sp:.0f}x", xy=(tc, cache_ttfts[i]),
                     xytext=(0, -20), textcoords="offset points",
                     fontsize=9, ha="center", color=CONDITION_COLORS["group_cached"],
                     fontweight="bold")

    ax1.set_xlabel("Number of Tools")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("TTFT Scaling")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cache storage cost
    ax2.bar(range(len(tool_counts)), cache_sizes, color=CONDITION_COLORS["group_cached"],
            alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(range(len(tool_counts)))
    ax2.set_xticklabels([str(tc) for tc in tool_counts])
    for i, (sz, tc) in enumerate(zip(cache_sizes, tool_counts)):
        ax2.text(i, sz + max(cache_sizes) * 0.02, f"{sz:.0f} MB",
                 ha="center", va="bottom", fontsize=10)
    ax2.set_xlabel("Number of Tools")
    ax2.set_ylabel("Cache Size (MB)")
    ax2.set_title("GPU Memory Cost")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig8_scaling_ttft.{ext}")
    print(f"  Saved fig8_scaling_ttft.pdf/png")
    plt.close(fig)


def generate_deployment_table(benchmark: dict, scaling_data: list[dict] | None, output_dir: Path):
    """Generate LaTeX deployment/amortization table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{ContextCache deployment economics. Compile cost is paid once when tools change; "
        r"every subsequent request benefits from cached KV states. TTFT = time to first token.}",
        r"\label{tab:deployment}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
    ]

    ch = benchmark["cache_hit"]
    fp = benchmark["full_prefill"]
    am = benchmark["amortization"]

    lines.append(f"  Tools in catalog & {benchmark['num_tools']} \\\\")
    lines.append(f"  Cache size (GPU) & {benchmark['cache_size_mb']:.0f} MB \\\\")
    lines.append(f"  Cached prefix tokens & {benchmark['cache_prefix_tokens']} \\\\")
    lines.append(r"  \midrule")
    lines.append(f"  Compile time (one-time) & {am['compile_cost_ms']:.0f} ms \\\\")
    lines.append(f"  Full prefill TTFT & {fp['ttft_ms']['mean']:.0f} ms \\\\")
    lines.append(f"  Cache-hit TTFT & {ch['ttft_ms']['mean']:.0f} ms \\\\")
    lines.append(f"  \\quad KV cache load & {ch['link_ms']['mean']:.0f} ms \\\\")
    lines.append(f"  \\quad Suffix prefill & {ch['prefill_ms']['mean']:.0f} ms \\\\")
    lines.append(r"  \midrule")
    lines.append(f"  TTFT speedup & {am['speedup_factor']:.1f}$\\times$ \\\\")
    lines.append(f"  Savings per request & {am['savings_per_request_ms']:.0f} ms \\\\")
    lines.append(f"  Break-even & {am['break_even_requests']:.1f} requests \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    # Also generate scaling table if available
    if scaling_data and len(scaling_data) >= 2:
        lines.append("")
        lines.append(r"% Scaling table")
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{TTFT scaling with number of tools. Full prefill grows linearly; "
                     r"ContextCache cache-hit TTFT remains constant (suffix-only prefill).}")
        lines.append(r"\label{tab:scaling}")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{rrrrrr}")
        lines.append(r"\toprule")
        lines.append(r"\# Tools & Full Prefill & Cache Hit & Speedup & Compile & Cache MB \\")
        lines.append(r"\midrule")

        for d in scaling_data:
            lines.append(
                f"  {d['num_tools']} "
                f"& {d['full_prefill']['ttft_ms']['mean']:.0f} ms "
                f"& {d['cache_hit']['ttft_ms']['mean']:.0f} ms "
                f"& {d['amortization']['speedup_factor']:.1f}$\\times$ "
                f"& {d['compile_ms']:.0f} ms "
                f"& {d['cache_size_mb']:.0f} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    out_path = output_dir / "table_deployment.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out_path}")


def print_summary(metrics: dict):
    """Print a readable summary table to stdout."""
    print(f"\n{'='*100}")
    print("CONTEXT CACHE EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"{'Split':<15} {'Condition':<13} {'TSA':>6} {'PF1':>6} {'EM':>6} "
          f"{'FPR':>6} {'FNR':>6} {'Link':>7} {'Prefill':>8} {'Decode':>8} {'N':>5}")
    print("-" * 100)

    for key in sorted(metrics.keys()):
        m = metrics[key]
        print(
            f"{m.get('split', '?'):<15} {m.get('condition', '?'):<13} "
            f"{m.get('tool_selection_accuracy', 0):>6.3f} "
            f"{m.get('parameter_f1', 0):>6.3f} "
            f"{m.get('exact_match', 0):>6.3f} "
            f"{m.get('false_positive_rate', 0):>6.3f} "
            f"{m.get('false_negative_rate', 0):>6.3f} "
            f"{m.get('avg_link_ms', 0):>7.0f} "
            f"{m.get('avg_prefill_ms', 0):>8.0f} "
            f"{m.get('avg_decode_ms', 0):>8.0f} "
            f"{m.get('num_examples', 0):>5}"
        )
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description="Generate ContextCache paper figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "eval_results" / "context_cache")
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "eval_results" / "context_cache_benchmark",
                        help="Path to benchmark_latency.py output")
    parser.add_argument("--gisting-dir", type=Path, default=None,
                        help="Path to gisting eval results for comparison")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "context_cache")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    metrics = load_summary(args.results_dir)
    if not metrics:
        print(f"  WARNING: No eval results found in {args.results_dir}")
    else:
        print(f"  Found {len(metrics)} result entries: {list(metrics.keys())}")
        print_summary(metrics)

    # Load benchmark results
    benchmark_main, benchmark_scaling = None, None
    if args.benchmark_dir and args.benchmark_dir.exists():
        benchmark_main, benchmark_scaling = load_benchmark(args.benchmark_dir)
        if benchmark_main:
            print(f"  Loaded benchmark: {benchmark_main['num_tools']} tools, {benchmark_main['num_queries']} queries")
        if benchmark_scaling:
            print(f"  Loaded scaling: {len(benchmark_scaling)} tool counts")

    # Load gisting results for comparison
    gisting_results = None
    if args.gisting_dir and args.gisting_dir.exists():
        summary_path = args.gisting_dir / "summary.json"
        rescored_path = args.gisting_dir / "summary_rescored.json"
        for p in (rescored_path, summary_path):
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    gisting_results = json.load(f)
                print(f"  Loaded gisting results from {p}")
                break

    # Generate quality/error analysis plots (from eval results)
    if metrics:
        print("\nGenerating quality/error analysis figures...")
        plot_quality_equivalence(metrics, args.output_dir)
        plot_latency_breakdown(metrics, args.output_dir)
        plot_error_analysis(metrics, args.output_dir)
        plot_split_comparison(metrics, args.output_dir)
        plot_method_comparison(metrics, gisting_results, args.output_dir)
        generate_latex_table(metrics, args.output_dir)

    # Generate deployment/amortization figures (from benchmark results)
    if benchmark_main:
        print("\nGenerating deployment/amortization figures...")
        plot_ttft_comparison(benchmark_main, args.output_dir)
        plot_amortization_curve(benchmark_main, args.output_dir)
        generate_deployment_table(benchmark_main, benchmark_scaling, args.output_dir)

    if benchmark_scaling:
        plot_scaling_ttft(benchmark_scaling, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
