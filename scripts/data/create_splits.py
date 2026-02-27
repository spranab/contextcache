#!/usr/bin/env python3
"""Create stratified train/val/test splits from formatted data."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def extract_tool_name(example: dict) -> str | None:
    """Extract the tool name from a formatted example."""
    for msg in example["messages"]:
        if msg["role"] == "assistant":
            content = msg["content"]
            if "<tool_call>" in content:
                try:
                    json_str = content.replace("<tool_call>", "").replace("</tool_call>", "")
                    call = json.loads(json_str)
                    return call.get("name")
                except json.JSONDecodeError:
                    pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "data_config.yaml",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "all_formatted.jsonl",
    )
    parser.add_argument(
        "--catalog-unseen",
        type=Path,
        default=ROOT / "data" / "catalogs" / "catalog_unseen.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    random.seed(args.seed)

    # Load all examples
    examples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} total examples")

    # Load unseen tool names
    unseen_tool_names = set()
    if args.catalog_unseen.exists():
        with open(args.catalog_unseen, encoding="utf-8") as f:
            unseen_catalog = json.load(f)
            unseen_tool_names = {
                t["schema"]["function"]["name"] for t in unseen_catalog
            }
    print(f"Unseen tools: {len(unseen_tool_names)}")

    # Group examples by tool name (None = non-tool)
    by_tool = defaultdict(list)
    for ex in examples:
        tool_name = extract_tool_name(ex)
        by_tool[tool_name].append(ex)

    # Separate unseen tool examples
    unseen_examples = []
    for tool_name in list(by_tool.keys()):
        if tool_name in unseen_tool_names:
            unseen_examples.extend(by_tool.pop(tool_name))

    print(f"Unseen tool examples: {len(unseen_examples)}")

    # Stratified split for seen tools + non-tool examples
    train_ratio = config["splits"]["train_ratio"]
    val_ratio = config["splits"]["val_ratio"]

    train, val, test_seen = [], [], []

    for tool_name, tool_examples in by_tool.items():
        random.shuffle(tool_examples)
        n = len(tool_examples)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train.extend(tool_examples[:n_train])
        val.extend(tool_examples[n_train : n_train + n_val])
        test_seen.extend(tool_examples[n_train + n_val :])

    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test_seen)
    random.shuffle(unseen_examples)

    # Remove the internal _is_tool tag before saving
    def clean(examples):
        for ex in examples:
            ex.pop("_is_tool", None)
        return examples

    train = clean(train)
    val = clean(val)
    test_seen = clean(test_seen)
    unseen_examples = clean(unseen_examples)

    # Save splits
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": train,
        "val": val,
        "test_seen": test_seen,
        "test_unseen": unseen_examples,
    }

    for name, data in splits.items():
        out_file = args.output_dir / f"{name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} examples -> {out_file}")

    # Print statistics
    print("\n--- Split Statistics ---")
    for name, data in splits.items():
        tool_count = sum(1 for ex in data if extract_tool_name(ex) is not None)
        non_tool_count = len(data) - tool_count
        tools_used = Counter(
            extract_tool_name(ex) for ex in data if extract_tool_name(ex) is not None
        )
        print(f"{name}:")
        print(f"  Total: {len(data)}")
        print(f"  Tool examples: {tool_count}")
        print(f"  Non-tool examples: {non_tool_count}")
        print(f"  Unique tools: {len(tools_used)}")
        if tools_used:
            print(f"  Min examples per tool: {min(tools_used.values())}")
            print(f"  Max examples per tool: {max(tools_used.values())}")

    # Save split stats
    stats = {}
    for name, data in splits.items():
        tools_used = Counter(
            extract_tool_name(ex) for ex in data if extract_tool_name(ex) is not None
        )
        stats[name] = {
            "total": len(data),
            "tool_examples": sum(
                1 for ex in data if extract_tool_name(ex) is not None
            ),
            "non_tool_examples": sum(
                1 for ex in data if extract_tool_name(ex) is None
            ),
            "unique_tools": len(tools_used),
        }

    stats_file = ROOT / "data" / "stats" / "split_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
