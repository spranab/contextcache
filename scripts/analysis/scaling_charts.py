#!/usr/bin/env python3
"""Generate publication-quality scaling charts from benchmark_scaling_100.json.

Creates 4 figures:
  1. TTFT vs Tool Count (cached vs full prefill) — the hero chart
  2. Speedup factor vs Tool Count
  3. Compile cost + break-even analysis
  4. Cache memory vs Tool Count

Usage:
    python scripts/analysis/scaling_charts.py
    python scripts/analysis/scaling_charts.py --input eval_results/context_cache_benchmark/benchmark_scaling_100.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colors
C_FULL = "#E74C3C"      # red for full prefill
C_CACHED = "#2ECC71"    # green for cached
C_SPEEDUP = "#3498DB"   # blue for speedup
C_COMPILE = "#F39C12"   # orange for compile
C_MEMORY = "#9B59B6"    # purple for memory


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def fig1_ttft_scaling(results: list[dict], output_dir: Path):
    """TTFT vs Tool Count — the hero chart."""
    tools = [r["num_tools"] for r in results]
    cached_mean = [r["cached_ttft"]["mean"] for r in results]
    cached_std = [r["cached_ttft"]["std"] for r in results]
    full_mean = [r["full_prefill_ttft"]["mean"] for r in results]
    full_std = [r["full_prefill_ttft"]["std"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Full prefill with error band
    ax.plot(tools, full_mean, "o-", color=C_FULL, linewidth=2.5,
            markersize=8, label="Full Prefill", zorder=3)
    ax.fill_between(tools,
                    [m - s for m, s in zip(full_mean, full_std)],
                    [m + s for m, s in zip(full_mean, full_std)],
                    color=C_FULL, alpha=0.15)

    # Cached with error band
    ax.plot(tools, cached_mean, "s-", color=C_CACHED, linewidth=2.5,
            markersize=8, label="ContextCache (group cached)", zorder=3)
    ax.fill_between(tools,
                    [m - s for m, s in zip(cached_mean, cached_std)],
                    [m + s for m, s in zip(cached_mean, cached_std)],
                    color=C_CACHED, alpha=0.15)

    # Annotate key points
    if len(results) >= 2:
        last = results[-1]
        speedup = last["speedup"]
        ax.annotate(
            f"{speedup:.1f}x faster",
            xy=(last["num_tools"], last["cached_ttft"]["mean"]),
            xytext=(last["num_tools"] - 15, last["cached_ttft"]["mean"] + 200),
            fontsize=12, fontweight="bold", color=C_CACHED,
            arrowprops=dict(arrowstyle="->", color=C_CACHED, lw=1.5),
        )

    ax.set_xlabel("Number of Tool Schemas")
    ax.set_ylabel("Time to First Token (ms)")
    ax.set_title("TTFT Scaling: ContextCache vs Full Prefill")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xticks(tools)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_dir / "scaling_ttft.png")
    fig.savefig(output_dir / "scaling_ttft.pdf")
    plt.close(fig)
    print(f"  -> scaling_ttft.png/pdf")


def fig2_speedup(results: list[dict], output_dir: Path):
    """Speedup factor vs Tool Count."""
    tools = [r["num_tools"] for r in results]
    speedups = [r["speedup"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(tools, speedups, width=[t * 0.15 for t in tools],
                  color=C_SPEEDUP, alpha=0.85, edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{sp:.1f}x", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=C_SPEEDUP)

    # Reference line at 1x
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="No speedup")

    ax.set_xlabel("Number of Tool Schemas")
    ax.set_ylabel("TTFT Speedup (x)")
    ax.set_title("Cache Speedup vs Tool Count")
    ax.set_xticks(tools)
    ax.set_ylim(bottom=0, top=max(speedups) * 1.25)

    fig.tight_layout()
    fig.savefig(output_dir / "scaling_speedup.png")
    fig.savefig(output_dir / "scaling_speedup.pdf")
    plt.close(fig)
    print(f"  -> scaling_speedup.png/pdf")


def fig3_compile_breakeven(results: list[dict], output_dir: Path):
    """Compile cost and break-even analysis (dual y-axis)."""
    tools = [r["num_tools"] for r in results]
    compile_ms = [r["compile_ms"] for r in results]
    breakeven = [r["break_even_requests"] for r in results]
    savings = [r["savings_per_request_ms"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Compile cost (left axis)
    line1 = ax1.plot(tools, compile_ms, "D-", color=C_COMPILE, linewidth=2.5,
                     markersize=8, label="Compile cost (ms)", zorder=3)
    ax1.set_ylabel("One-time Compile Cost (ms)", color=C_COMPILE)
    ax1.tick_params(axis="y", labelcolor=C_COMPILE)

    # Savings per request (right axis)
    line2 = ax2.plot(tools, savings, "^-", color=C_CACHED, linewidth=2.5,
                     markersize=8, label="Savings per request (ms)", zorder=3)
    ax2.set_ylabel("Savings per Request (ms)", color=C_CACHED)
    ax2.tick_params(axis="y", labelcolor=C_CACHED)

    # Break-even annotations
    for i, (t, be) in enumerate(zip(tools, breakeven)):
        ax1.annotate(f"BE: {be:.1f} req",
                     xy=(t, compile_ms[i]),
                     xytext=(0, 12), textcoords="offset points",
                     fontsize=9, ha="center", color="#555")

    ax1.set_xlabel("Number of Tool Schemas")
    ax1.set_title("Compile Cost & Per-Request Savings")
    ax1.set_xticks(tools)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_dir / "scaling_compile_savings.png")
    fig.savefig(output_dir / "scaling_compile_savings.pdf")
    plt.close(fig)
    print(f"  -> scaling_compile_savings.png/pdf")


def fig4_memory(results: list[dict], output_dir: Path):
    """Cache memory usage vs Tool Count."""
    tools = [r["num_tools"] for r in results]
    memory = [r["cache_size_mb"] for r in results]
    tokens = [r["prefix_tokens"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(tools, memory, width=[t * 0.15 for t in tools],
                   color=C_MEMORY, alpha=0.8, edgecolor="white", linewidth=1.5,
                   label="Cache size (MB)")
    ax1.set_ylabel("GPU Cache Size (MB)", color=C_MEMORY)
    ax1.tick_params(axis="y", labelcolor=C_MEMORY)

    line = ax2.plot(tools, tokens, "o--", color="#555", linewidth=1.5,
                    markersize=6, label="Prefix tokens")
    ax2.set_ylabel("Prefix Tokens", color="#555")
    ax2.tick_params(axis="y", labelcolor="#555")

    # MB per tool annotation
    for bar, mem, n in zip(bars, memory, tools):
        per_tool = mem / n
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{per_tool:.0f}\nMB/tool", ha="center", va="bottom",
                 fontsize=8, color=C_MEMORY)

    ax1.set_xlabel("Number of Tool Schemas")
    ax1.set_title("Cache Memory & Token Scaling")
    ax1.set_xticks(tools)
    ax1.set_ylim(bottom=0)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_dir / "scaling_memory.png")
    fig.savefig(output_dir / "scaling_memory.pdf")
    plt.close(fig)
    print(f"  -> scaling_memory.png/pdf")


def fig5_combined_hero(results: list[dict], output_dir: Path):
    """2x2 combined figure for presentations."""
    tools = [r["num_tools"] for r in results]
    cached_mean = [r["cached_ttft"]["mean"] for r in results]
    full_mean = [r["full_prefill_ttft"]["mean"] for r in results]
    speedups = [r["speedup"] for r in results]
    compile_ms = [r["compile_ms"] for r in results]
    savings = [r["savings_per_request_ms"] for r in results]
    memory = [r["cache_size_mb"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) TTFT scaling
    ax = axes[0, 0]
    ax.plot(tools, full_mean, "o-", color=C_FULL, linewidth=2, markersize=7, label="Full Prefill")
    ax.plot(tools, cached_mean, "s-", color=C_CACHED, linewidth=2, markersize=7, label="ContextCache")
    ax.set_xlabel("Tools")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("(a) Time to First Token")
    ax.legend(loc="upper left")
    ax.set_xticks(tools)
    ax.set_ylim(bottom=0)

    # (b) Speedup
    ax = axes[0, 1]
    ax.bar(tools, speedups, width=[t * 0.15 for t in tools],
           color=C_SPEEDUP, alpha=0.85, edgecolor="white")
    for t, sp in zip(tools, speedups):
        ax.text(t, sp + 0.15, f"{sp:.1f}x", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Tools")
    ax.set_ylabel("Speedup (x)")
    ax.set_title("(b) TTFT Speedup")
    ax.set_xticks(tools)
    ax.set_ylim(bottom=0)

    # (c) Economics: compile + savings
    ax = axes[1, 0]
    ax.plot(tools, compile_ms, "D-", color=C_COMPILE, linewidth=2, markersize=7, label="Compile cost")
    ax.plot(tools, savings, "^-", color=C_CACHED, linewidth=2, markersize=7, label="Savings/request")
    ax.set_xlabel("Tools")
    ax.set_ylabel("Time (ms)")
    ax.set_title("(c) Compile Cost vs Savings")
    ax.legend(loc="upper left")
    ax.set_xticks(tools)

    # (d) Memory
    ax = axes[1, 1]
    ax.bar(tools, memory, width=[t * 0.15 for t in tools],
           color=C_MEMORY, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Tools")
    ax.set_ylabel("Cache (MB)")
    ax.set_title("(d) GPU Cache Size")
    ax.set_xticks(tools)
    ax.set_ylim(bottom=0)

    fig.suptitle("ContextCache Scaling: 5 to 100 Tools (Qwen3-8B, 4-bit NF4, RTX 3090 Ti)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "scaling_combined.png")
    fig.savefig(output_dir / "scaling_combined.pdf")
    plt.close(fig)
    print(f"  -> scaling_combined.png/pdf")


def print_latex_table(results: list[dict]):
    """Print LaTeX table for the paper."""
    print("\n% LaTeX table")
    print("\\begin{tabular}{rrrrrrrr}")
    print("\\toprule")
    print("Tools & Tokens & Cache (MB) & Compile (ms) & Cached TTFT & Full TTFT & Speedup & Break-even \\\\")
    print("\\midrule")
    for r in results:
        print(f"{r['num_tools']} & {r['prefix_tokens']:,} & {r['cache_size_mb']:.0f} & "
              f"{r['compile_ms']:.0f} & {r['cached_ttft']['mean']:.0f} & "
              f"{r['full_prefill_ttft']['mean']:.0f} & {r['speedup']:.1f}$\\times$ & "
              f"{r['break_even_requests']:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    parser = argparse.ArgumentParser(description="Generate scaling charts")
    parser.add_argument("--input", type=str,
                        default="eval_results/context_cache_benchmark/benchmark_scaling_100.json")
    parser.add_argument("--output-dir", type=str, default="figures/context_cache")
    args = parser.parse_args()

    results = load_results(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(results)} data points from {args.input}")
    print(f"Tool counts: {[r['num_tools'] for r in results]}")
    print(f"Generating charts to {output_dir}/\n")

    fig1_ttft_scaling(results, output_dir)
    fig2_speedup(results, output_dir)
    fig3_compile_breakeven(results, output_dir)
    fig4_memory(results, output_dir)
    fig5_combined_hero(results, output_dir)
    print_latex_table(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
