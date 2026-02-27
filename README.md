# ContextCache

**Persistent KV Cache with Content-Hash Addressing for Zero-Degradation Tool Schema Caching**

ContextCache accelerates tool-augmented LLM inference by persistently caching the KV states of tool schema prefixes. On cache hits, only the user query (suffix) needs prefilling, reducing time-to-first-token (TTFT) by up to **6.9x** with **zero quality degradation** compared to full prefill.

The system uses SHA-256 content-hash addressing for cache keys, supports disk persistence across server restarts, and includes a production-ready FastAPI serving layer with a browser-based UI.

## Key Results

### Quality: Zero Degradation

| Split | Condition | TSA | PF1 | EM | FPR | FNR |
|-------|-----------|-----|-----|----|-----|-----|
| test_seen | group_cached | **0.850** | 0.735 | 0.600 | 0.000 | 0.050 |
| test_seen | full_prefill | 0.850 | 0.716 | 0.550 | 0.000 | 0.050 |
| test_held_out | group_cached | **0.900** | 0.694 | 0.600 | 0.000 | 0.100 |
| test_held_out | full_prefill | 0.900 | 0.694 | 0.600 | 0.000 | 0.100 |
| test_unseen | group_cached | **0.850** | 0.676 | 0.550 | 0.000 | 0.150 |
| test_unseen | full_prefill | 0.850 | 0.676 | 0.550 | 0.000 | 0.150 |

Group-cached matches full prefill **exactly** on TSA across all 3 splits.

### Latency: 6.9x TTFT Speedup

| Metric | Value |
|--------|-------|
| Tools in catalog | 20 |
| Cache size (GPU) | 401 MB |
| Compile time (one-time) | 1,386 ms |
| Full prefill TTFT | 787 ms |
| Cache-hit TTFT | 114 ms |
| **TTFT speedup** | **6.9x** |
| Break-even | 2.1 requests |

### Scaling

| # Tools | Full Prefill | Cache Hit | Speedup |
|---------|-------------|-----------|---------|
| 5 | 237 ms | 120 ms | 2.0x |
| 10 | 395 ms | 114 ms | 3.5x |
| 20 | 794 ms | 121 ms | 6.6x |

Cache-hit TTFT stays constant as tool count grows (suffix-only prefill).

## Architecture

```
Tool Schemas (JSON)
  ↓ SHA-256 hash
[Content-Hash Cache Key]
  ↓
[Group KV Compilation] → system + tools prefilled together
  ↓ Store to disk
[Persistent KV Store] → .safetensors files
  ↓ Load on cache hit
[Suffix-Only Prefill] → only user query needs forward pass
  ↓
[Generation]
```

**Key insight**: Per-tool independent KV compilation fails (TSA ~0.1) because tool tokens need cross-attention to the system prompt and each other during prefill. Group caching (system + all tools together) preserves these dependencies and matches full prefill exactly.

## Quick Start

### Install

```bash
git clone https://github.com/spranab/contextcache.git
cd contextcache
pip install -r requirements.txt
```

### One-Command Server Launch

```bash
# Live mode (requires GPU with ~8GB VRAM)
python scripts/serve/launch.py

# Demo mode (no GPU needed — pre-recorded responses)
python scripts/serve/launch.py --demo
```

**Live mode** starts the FastAPI server, loads Qwen3-8B (4-bit NF4) in the background, and opens a browser UI.

**Demo mode** runs without any GPU or model dependencies, using pre-recorded responses with realistic timing simulation. Ideal for presentations and evaluating the UI.

### Programmatic Usage

```python
from context_cache import ContextCache, CacheConfig

config = CacheConfig(model_name="Qwen/Qwen3-8B", cache_dir="./cache")
cc = ContextCache(config)

# Register tools (compiles and caches KV states)
cc.register_tools([
    {"type": "function", "function": {"name": "get_weather", ...}},
    {"type": "function", "function": {"name": "search_web", ...}},
])

# Query with cached KV (suffix-only prefill)
response = cc.query("What's the weather in Tokyo?")
```

### Evaluation

```bash
# Quality evaluation (TSA, PF1, EM across 3 conditions × 3 splits)
python scripts/eval/eval_context_cache.py --config configs/context_cache_config.yaml

# Latency benchmarks
python scripts/cache/benchmark_latency.py
```

### Data Preparation

```bash
python scripts/data/download_sources.py
python scripts/data/build_catalog.py
python scripts/data/create_splits.py
python scripts/data/format_training_data.py
```

## Project Structure

```
context_cache/                    # Core package
  context_cache.py                # Main cache engine
  cache_config.py                 # Configuration
  kv_store.py                     # Disk-persistent KV storage
  model_adapter.py                # Model-agnostic adapter (Qwen, Llama, Mistral)
  rope_utils.py                   # NoPE capture & deferred RoPE
configs/
  context_cache_config.yaml       # Evaluation configuration
scripts/
  eval/eval_context_cache.py      # Quality evaluation
  cache/benchmark_latency.py      # Latency benchmarks
  cache/compile_contexts.py       # Pre-compile KV caches
  cache/smoke_test.py             # Quick validation
  serve/serve_context_cache.py    # FastAPI server (live + demo modes)
  serve/launch.py                 # One-command launcher (--demo flag)
  serve/demo_profiles.json        # Team profiles (Customer Service, Inventory, Analytics)
  serve/demo_recording.json       # Pre-recorded responses + timing profiles
  serve/test_client.py            # API test client
  serve/sample_tools.json         # Demo tools
  serve/static/index.html         # Browser UI with dashboard
  data/                           # Data pipeline
  analysis/cache_plots.py         # Plot generation
eval_results/                     # Paper-ready results
figures/                          # Paper figures (PDF + PNG)
paper/                            # LaTeX source & compiled PDF
```

## Serving API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools` | POST | Register tool schemas (triggers KV compilation) |
| `/query` | POST | Query with cached context |
| `/tools` | DELETE | Clear registered tools and cache |
| `/status` | GET | Cache stats (tool count, size, model info) |
| `/health` | GET | Health check (model loaded status) |
| `/sample-tools` | GET | Demo tool schemas |
| `/profiles` | GET | List available team profiles (demo mode) |
| `/profiles/{id}/tools` | GET | Get tools for a specific profile |
| `/query/compare` | POST | A/B comparison (cached vs full prefill) |
| `/mode` | GET | Current server mode (live or demo) |

## Paper

**ContextCache: Persistent KV Cache with Content-Hash Addressing for Zero-Degradation Tool Schema Caching**
Pranab Sarkar, 2026

Paper PDF: [paper/main.pdf](paper/main.pdf)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18795189.svg)](https://doi.org/10.5281/zenodo.18795189)

## Citation

```bibtex
@techreport{sarkar2026contextcache,
  title={ContextCache: Persistent {KV} Cache with Content-Hash Addressing for Zero-Degradation Tool Schema Caching},
  author={Sarkar, Pranab},
  year={2026},
  institution={Zenodo},
  doi={10.5281/zenodo.18795189},
  url={https://doi.org/10.5281/zenodo.18795189}
}
```

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
