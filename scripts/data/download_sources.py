#!/usr/bin/env python3
"""Download source datasets from HuggingFace for tool catalog construction."""

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def download_dataset(name: str, hf_path: str, output_dir: Path) -> int:
    """Download a single dataset and save to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Downloading: {name} ({hf_path})")
    print(f"{'='*60}")

    try:
        ds = load_dataset(hf_path)
    except Exception as e:
        print(f"  [ERROR] Failed to download {hf_path}: {e}")
        return 0

    total = 0
    for split_name, split_data in ds.items():
        out_file = output_dir / f"{split_name}.jsonl"
        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for item in tqdm(split_data, desc=f"  {split_name}"):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
        print(f"  {split_name}: {count} examples -> {out_file}")
        total += count

    return total


def main():
    parser = argparse.ArgumentParser(description="Download source datasets")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "data_config.yaml",
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Download only these sources (by name). Default: all.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sources = config["sources"]
    if args.sources:
        sources = [s for s in sources if s["name"] in args.sources]

    print(f"Will download {len(sources)} dataset(s) to {RAW_DIR}")

    grand_total = 0
    for source in sources:
        output_dir = RAW_DIR / source["name"]
        count = download_dataset(source["name"], source["hf_path"], output_dir)
        grand_total += count

    print(f"\n{'='*60}")
    print(f"Done. Total examples downloaded: {grand_total}")
    print(f"Data saved to: {RAW_DIR}")


if __name__ == "__main__":
    main()
