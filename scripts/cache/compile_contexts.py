#!/usr/bin/env python3
"""Compile tool schemas (or any context blocks) into NoPE KV cache.

Usage:
    python scripts/cache/compile_contexts.py --config configs/context_cache_config.yaml
    python scripts/cache/compile_contexts.py --config configs/context_cache_config.yaml --catalog data/catalogs/catalog_100.json
    python scripts/cache/compile_contexts.py --config configs/context_cache_config.yaml --files src/*.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from context_cache.cache_config import ContextCacheConfig
from context_cache.context_cache import ContextCacheModel


def compile_tool_catalogs(model: ContextCacheModel, catalog_paths: list[str]):
    """Compile all tools from one or more catalog JSON files."""
    total_tools = 0
    total_tokens = 0

    for catalog_path in catalog_paths:
        print(f"\nCompiling catalog: {catalog_path}")
        with open(catalog_path) as f:
            catalog = json.load(f)

        for tool in catalog:
            schema = tool.get("schema", tool)
            name = schema.get("function", {}).get("name", tool.get("tool_id", "unknown"))
            schema_text = json.dumps(schema, separators=(",", ":"))

            t0 = time.perf_counter()
            cached = model.compile_context(schema_text, name, content_type="tool_schema")
            elapsed = (time.perf_counter() - t0) * 1000

            print(f"  {name}: {cached.num_tokens} tokens, {elapsed:.0f}ms")
            total_tools += 1
            total_tokens += cached.num_tokens

    return total_tools, total_tokens


def compile_files(model: ContextCacheModel, file_paths: list[str]):
    """Compile code files or documents."""
    total_files = 0
    total_tokens = 0

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"  Skipping (not found): {file_path}")
            continue

        content = path.read_text(encoding="utf-8", errors="replace")
        name = str(path)

        t0 = time.perf_counter()
        cached = model.compile_context(content, name, content_type="code_file")
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"  {name}: {cached.num_tokens} tokens, {elapsed:.0f}ms")
        total_files += 1
        total_tokens += cached.num_tokens

    return total_files, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Compile context blocks into NoPE KV cache")
    parser.add_argument("--config", type=str, default="configs/context_cache_config.yaml")
    parser.add_argument("--catalog", type=str, nargs="*", help="Tool catalog JSON files")
    parser.add_argument("--files", type=str, nargs="*", help="Code/document files to compile")
    parser.add_argument("--force", action="store_true", help="Recompile even if cached")
    args = parser.parse_args()

    config = ContextCacheConfig.from_yaml(args.config)

    # Override force flag
    if args.force:
        print("Force recompilation enabled")

    model = ContextCacheModel(config)

    total_entries = 0
    total_tokens = 0

    # Compile tool catalogs
    catalog_paths = args.catalog or config.eval.catalogs
    if catalog_paths:
        n, t = compile_tool_catalogs(model, catalog_paths)
        total_entries += n
        total_tokens += t

    # Compile files
    if args.files:
        print(f"\nCompiling {len(args.files)} files...")
        n, t = compile_files(model, args.files)
        total_entries += n
        total_tokens += t

    # Print summary
    print(f"\n{'='*60}")
    print(f"Compilation complete:")
    print(f"  Entries compiled: {total_entries}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg tokens/entry: {total_tokens / max(total_entries, 1):.0f}")
    print(f"  Cache entries on disk: {len(model.kv_store)}")

    # Print cache size
    cache_dir = Path(config.cache.cache_dir)
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.glob("*.pt"))
        print(f"  Cache size on disk: {total_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
