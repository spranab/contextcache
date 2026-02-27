#!/usr/bin/env python3
"""Build a curated tool catalog from downloaded source datasets.

Extracts unique tool schemas from xLAM, Hermes, Glaive, and Gorilla datasets,
deduplicates them, categorizes by domain, and produces catalog JSON files
for the 50/100/200-tool scaling variants.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CATALOG_DIR = ROOT / "data" / "catalogs"
STATS_DIR = ROOT / "data" / "stats"

# Domain classification keywords (heuristic)
DOMAIN_KEYWORDS = {
    "weather": ["weather", "temperature", "forecast", "climate", "humidity", "wind"],
    "finance": ["stock", "price", "market", "currency", "exchange", "portfolio", "trade", "crypto"],
    "search": ["search", "query", "find", "lookup", "browse", "google"],
    "calendar": ["calendar", "event", "schedule", "meeting", "appointment", "reminder"],
    "email": ["email", "mail", "send_email", "inbox", "message"],
    "file_management": ["file", "directory", "folder", "read_file", "write_file", "delete_file", "list_files"],
    "database": ["database", "sql", "query_db", "insert", "update_record", "delete_record"],
    "communication": ["slack", "notification", "notify", "chat", "sms", "call"],
    "math": ["calculate", "math", "compute", "convert", "unit"],
    "web": ["http", "api", "request", "fetch", "url", "download", "scrape"],
    "utilities": ["timestamp", "random", "uuid", "hash", "encode", "decode", "format"],
    "social_media": ["tweet", "post", "social", "instagram", "facebook", "linkedin"],
    "ecommerce": ["order", "cart", "product", "payment", "shipping", "inventory"],
    "location": ["location", "map", "geocode", "address", "distance", "route", "gps"],
    "media": ["image", "video", "audio", "photo", "music", "play"],
}


def classify_domain(tool_name: str, description: str) -> str:
    """Classify a tool into a domain based on name and description."""
    text = f"{tool_name} {description}".lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def normalize_schema(schema: dict) -> dict:
    """Normalize a tool schema to a consistent format."""
    # Handle different source formats
    if "function" in schema:
        func = schema["function"]
    elif "name" in schema and "parameters" in schema:
        func = schema
    else:
        return None

    name = func.get("name", "").strip()
    if not name:
        return None

    description = func.get("description", "").strip()
    parameters = func.get("parameters", {})

    # Ensure parameters has the right structure
    if not isinstance(parameters, dict):
        parameters = {}
    if "type" not in parameters:
        parameters["type"] = "object"
    if "properties" not in parameters:
        parameters["properties"] = {}

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def extract_tools_xlam(raw_dir: Path) -> list[dict]:
    """Extract tool schemas from Salesforce xLAM dataset."""
    tools = []
    data_file = raw_dir / "xlam" / "train.jsonl"
    if not data_file.exists():
        print(f"  [SKIP] {data_file} not found")
        return tools

    seen_names = set()
    with open(data_file, encoding="utf-8") as f:
        for line in tqdm(f, desc="  xLAM"):
            item = json.loads(line)
            # xLAM stores tools as a JSON string in the "tools" field
            tools_raw = item.get("tools")
            if isinstance(tools_raw, str):
                try:
                    tools_list = json.loads(tools_raw)
                except json.JSONDecodeError:
                    continue
            elif isinstance(tools_raw, list):
                tools_list = tools_raw
            else:
                continue

            for t in tools_list:
                schema = normalize_schema(t)
                if schema and schema["function"]["name"] not in seen_names:
                    seen_names.add(schema["function"]["name"])
                    tools.append(schema)

    return tools


def extract_tools_hermes(raw_dir: Path) -> list[dict]:
    """Extract tool schemas from NousResearch Hermes dataset."""
    tools = []
    seen_names = set()

    for jsonl_file in sorted(raw_dir.glob("hermes/*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in tqdm(f, desc=f"  Hermes ({jsonl_file.name})"):
                item = json.loads(line)
                # Hermes uses ShareGPT format with "conversations"
                conversations = item.get("conversations", [])
                for msg in conversations:
                    if msg.get("from") == "system":
                        # Try to extract tool definitions from system message
                        content = msg.get("value", "")
                        # Look for JSON tool definitions
                        try:
                            # Some Hermes examples have tools in a JSON block
                            matches = re.findall(
                                r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*"parameters"\s*:[^{}]*\{[^}]*\}[^{}]*\}',
                                content,
                                re.DOTALL,
                            )
                            for m in matches:
                                try:
                                    parsed = json.loads(m)
                                    schema = normalize_schema(parsed)
                                    if schema and schema["function"]["name"] not in seen_names:
                                        seen_names.add(schema["function"]["name"])
                                        tools.append(schema)
                                except json.JSONDecodeError:
                                    pass
                        except Exception:
                            pass

    return tools


def extract_tools_glaive(raw_dir: Path) -> list[dict]:
    """Extract tool schemas from Glaive Function Calling v2."""
    tools = []
    seen_names = set()

    for jsonl_file in sorted(raw_dir.glob("glaive/*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in tqdm(f, desc=f"  Glaive ({jsonl_file.name})"):
                item = json.loads(line)
                # Glaive has a "system" field with tool definitions
                system = item.get("system", "")
                # Extract function definitions
                func_matches = re.findall(
                    r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"description"\s*:\s*"([^"]*)"[^{}]*"parameters"\s*:\s*(\{[^}]*\})',
                    system,
                    re.DOTALL,
                )
                for name, desc, params_str in func_matches:
                    if name not in seen_names:
                        try:
                            params = json.loads(params_str)
                        except json.JSONDecodeError:
                            params = {"type": "object", "properties": {}}
                        schema = normalize_schema(
                            {"name": name, "description": desc, "parameters": params}
                        )
                        if schema:
                            seen_names.add(name)
                            tools.append(schema)

    return tools


def deduplicate_tools(tools: list[dict], threshold: float = 0.95) -> list[dict]:
    """Deduplicate tools by name similarity and optionally by description embedding."""
    seen = {}
    unique = []
    for tool in tools:
        name = tool["function"]["name"].lower().strip()
        # Normalize name: replace hyphens/dots with underscores
        norm_name = re.sub(r"[-.]", "_", name)
        if norm_name not in seen:
            seen[norm_name] = tool
            unique.append(tool)
    return unique


def compute_tool_id(schema: dict) -> str:
    """Generate a deterministic ID for a tool schema."""
    content = json.dumps(schema, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def build_catalog(tools: list[dict], num_tools: int) -> list[dict]:
    """Select tools for a catalog, ensuring domain diversity."""
    # Classify all tools by domain
    by_domain = defaultdict(list)
    for tool in tools:
        func = tool["function"]
        domain = classify_domain(func["name"], func["description"])
        tool_entry = {
            "tool_id": compute_tool_id(tool),
            "domain": domain,
            "schema": tool,
            "param_count": len(
                tool["function"].get("parameters", {}).get("properties", {})
            ),
        }
        by_domain[domain].append(tool_entry)

    # Round-robin selection across domains for diversity
    catalog = []
    domains = sorted(by_domain.keys())
    idx = {d: 0 for d in domains}

    while len(catalog) < num_tools and any(
        idx[d] < len(by_domain[d]) for d in domains
    ):
        for domain in domains:
            if len(catalog) >= num_tools:
                break
            if idx[domain] < len(by_domain[domain]):
                catalog.append(by_domain[domain][idx[domain]])
                idx[domain] += 1

    return catalog


def main():
    parser = argparse.ArgumentParser(description="Build tool catalogs from sources")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "data_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    # Extract tools from each source
    print("Extracting tool schemas from source datasets...")
    all_tools = []

    extractors = {
        "xlam": extract_tools_xlam,
        "hermes": extract_tools_hermes,
        "glaive": extract_tools_glaive,
    }

    for name, extractor in extractors.items():
        print(f"\n[{name}]")
        tools = extractor(RAW_DIR)
        print(f"  Extracted {len(tools)} unique tools")
        all_tools.extend(tools)

    print(f"\nTotal tools before dedup: {len(all_tools)}")

    # Deduplicate
    unique_tools = deduplicate_tools(all_tools)
    print(f"Total tools after dedup: {len(unique_tools)}")

    # Build catalog variants
    for size in config["catalog"]["scaling_variants"]:
        catalog = build_catalog(unique_tools, size)
        out_file = CATALOG_DIR / f"catalog_{size}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"\nCatalog ({size} tools) saved to {out_file}")

        # Stats
        domains = Counter(t["domain"] for t in catalog)
        param_counts = [t["param_count"] for t in catalog]
        print(f"  Domains: {dict(domains)}")
        print(f"  Param count range: {min(param_counts)}-{max(param_counts)}")

    # Also create the unseen tools set (tools NOT in the primary catalog)
    primary_catalog = json.loads(
        (CATALOG_DIR / f"catalog_{config['catalog']['num_seen_tools']}.json").read_text()
    )
    primary_ids = {t["tool_id"] for t in primary_catalog}

    remaining = [t for t in unique_tools if compute_tool_id(t) not in primary_ids]
    unseen = build_catalog(remaining, config["catalog"]["num_unseen_tools"])
    unseen_file = CATALOG_DIR / "catalog_unseen.json"
    with open(unseen_file, "w", encoding="utf-8") as f:
        json.dump(unseen, f, indent=2, ensure_ascii=False)
    print(f"\nUnseen tools ({len(unseen)}) saved to {unseen_file}")

    # Save stats
    stats = {
        "total_extracted": len(all_tools),
        "total_unique": len(unique_tools),
        "catalogs": {
            str(size): {
                "count": len(
                    json.loads((CATALOG_DIR / f"catalog_{size}.json").read_text())
                ),
            }
            for size in config["catalog"]["scaling_variants"]
        },
        "unseen_count": len(unseen),
    }
    with open(STATS_DIR / "catalog_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
