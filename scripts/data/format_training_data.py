#!/usr/bin/env python3
"""Format training data into Qwen 3 ChatML format WITHOUT tool schemas.

This is the critical script that creates the teacher-student asymmetry:
- Source data was generated/collected WITH tool schemas visible
- Training data is formatted WITHOUT tool schemas in the prompt
- The model must learn to call tools from the query alone
"""

import argparse
import json
import re
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

# System prompt for the distilled model (NO tool schemas)
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a set of tools. "
    "When the user's request requires a tool, respond with the appropriate "
    "tool call. When no tool is needed, respond normally."
)

# System prompt template for schema-included conditions (used in evaluation only)
SYSTEM_PROMPT_WITH_SCHEMA = (
    "You are a helpful assistant with access to the following tools:\n\n"
    "{tools_json}\n\n"
    "When the user's request requires a tool, respond with the appropriate "
    "tool call. When no tool is needed, respond normally."
)


def format_tool_call_qwen3(tool_call: dict) -> str:
    """Format a tool call in Qwen 3's native format."""
    return f"<tool_call>{json.dumps(tool_call, ensure_ascii=False)}</tool_call>"


def format_example_no_schema(query: str, tool_call: dict | None, response: str | None = None) -> dict:
    """Format a single example in Qwen 3 ChatML WITHOUT tool schemas.

    This is the training format for our method (condition c).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    if tool_call is not None:
        assistant_content = format_tool_call_qwen3(tool_call)
    else:
        assistant_content = response or "I can help with that."

    messages.append({"role": "assistant", "content": assistant_content})

    return {"messages": messages}


def extract_from_xlam(raw_dir: Path, catalog_tools: set[str]) -> list[dict]:
    """Extract matching examples from xLAM dataset."""
    examples = []
    data_file = raw_dir / "xlam" / "train.jsonl"
    if not data_file.exists():
        return examples

    with open(data_file, encoding="utf-8") as f:
        for line in tqdm(f, desc="  xLAM"):
            item = json.loads(line)
            query = item.get("query", "").strip()
            if not query:
                continue

            # Parse tool calls from the answer
            answers_raw = item.get("answers")
            if isinstance(answers_raw, str):
                try:
                    answers = json.loads(answers_raw)
                except json.JSONDecodeError:
                    continue
            elif isinstance(answers_raw, list):
                answers = answers_raw
            else:
                continue

            for answer in answers:
                if isinstance(answer, dict) and "name" in answer:
                    if answer["name"] in catalog_tools:
                        tool_call = {
                            "name": answer["name"],
                            "arguments": answer.get("arguments", {}),
                        }
                        examples.append(
                            format_example_no_schema(query, tool_call)
                        )

    return examples


def extract_from_glaive(raw_dir: Path, catalog_tools: set[str]) -> list[dict]:
    """Extract matching examples from Glaive dataset."""
    examples = []

    for jsonl_file in sorted(raw_dir.glob("glaive/*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in tqdm(f, desc=f"  Glaive ({jsonl_file.name})"):
                item = json.loads(line)
                chat = item.get("chat", "")

                # Parse chat format: USER: ... ASSISTANT: ... FUNCTION CALL: ...
                user_match = re.search(r"USER:\s*(.*?)(?=ASSISTANT:|FUNCTION|$)", chat, re.DOTALL)
                func_match = re.search(
                    r"FUNCTION CALL:\s*(\{.*?\})", chat, re.DOTALL
                )

                if user_match and func_match:
                    query = user_match.group(1).strip()
                    try:
                        func_call = json.loads(func_match.group(1))
                        if isinstance(func_call, dict) and func_call.get("name") in catalog_tools:
                            tool_call = {
                                "name": func_call["name"],
                                "arguments": func_call.get("arguments", func_call.get("parameters", {})),
                            }
                            examples.append(
                                format_example_no_schema(query, tool_call)
                            )
                    except json.JSONDecodeError:
                        pass
                elif user_match and not func_match:
                    # This is a non-tool example from Glaive
                    query = user_match.group(1).strip()
                    assistant_match = re.search(
                        r"ASSISTANT:\s*(.*?)(?=USER:|FUNCTION|$)", chat, re.DOTALL
                    )
                    if assistant_match and query:
                        response = assistant_match.group(1).strip()
                        if response and len(response) > 10:
                            examples.append(
                                format_example_no_schema(query, None, response)
                            )

    return examples


def load_synthetic_examples(synthetic_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load synthetic tool and non-tool examples."""
    tool_examples = []
    non_tool_examples = []

    tool_file = synthetic_dir / "raw_generations" / "tool_examples.jsonl"
    if tool_file.exists():
        with open(tool_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("tool_call"):
                    formatted = format_example_no_schema(
                        item["query"], item["tool_call"]
                    )
                    tool_examples.append(formatted)

    non_tool_file = synthetic_dir / "raw_generations" / "non_tool_examples.jsonl"
    if non_tool_file.exists():
        with open(non_tool_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                formatted = format_example_no_schema(
                    item["query"], None, item.get("response", "I'd be happy to help.")
                )
                non_tool_examples.append(formatted)

    return tool_examples, non_tool_examples


def main():
    parser = argparse.ArgumentParser(
        description="Format training data in Qwen 3 ChatML"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "data_config.yaml",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "data" / "catalogs" / "catalog_100.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(args.catalog, encoding="utf-8") as f:
        catalog = json.load(f)

    catalog_tools = {t["schema"]["function"]["name"] for t in catalog}
    print(f"Catalog has {len(catalog_tools)} tools")

    raw_dir = ROOT / "data" / "raw"
    synthetic_dir = ROOT / "data" / "synthetic"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Extract from existing datasets
    print("\nExtracting from existing datasets...")
    existing_examples = []

    xlam_examples = extract_from_xlam(raw_dir, catalog_tools)
    print(f"  xLAM: {len(xlam_examples)} examples")
    existing_examples.extend(xlam_examples)

    glaive_examples = extract_from_glaive(raw_dir, catalog_tools)
    print(f"  Glaive: {len(glaive_examples)} examples")
    existing_examples.extend(glaive_examples)

    # Load synthetic examples
    print("\nLoading synthetic examples...")
    synth_tool, synth_non_tool = load_synthetic_examples(synthetic_dir)
    print(f"  Synthetic tool: {len(synth_tool)} examples")
    print(f"  Synthetic non-tool: {len(synth_non_tool)} examples")

    # Combine all examples
    tool_examples = [
        ex for ex in existing_examples
        if any(
            "<tool_call>" in msg.get("content", "")
            for msg in ex["messages"]
            if msg["role"] == "assistant"
        )
    ]
    non_tool_examples = [
        ex for ex in existing_examples
        if not any(
            "<tool_call>" in msg.get("content", "")
            for msg in ex["messages"]
            if msg["role"] == "assistant"
        )
    ]

    tool_examples.extend(synth_tool)
    non_tool_examples.extend(synth_non_tool)

    print(f"\nTotal tool examples: {len(tool_examples)}")
    print(f"Total non-tool examples: {len(non_tool_examples)}")

    # ── Curriculum ordering ──────────────────────────────────────────────
    # Strategy: stratified sampling that ensures early exposure to all tools,
    # then ramps difficulty. This is simpler and more robust than temporal
    # ordering and avoids forcing unnatural distributions.
    #
    # 1. Group tool examples by tool name
    # 2. Round-robin one example per tool (exposure phase)
    # 3. Shuffle remaining examples
    # 4. Interleave non-tool examples throughout
    print("\nApplying curriculum ordering...")

    # Tag tool examples with their tool name for grouping
    tool_to_examples: dict[str, list[dict]] = {}
    for ex in tool_examples:
        name = None
        for msg in ex["messages"]:
            if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
                try:
                    call = json.loads(
                        msg["content"].replace("<tool_call>", "").replace("</tool_call>", "")
                    )
                    name = call.get("name", "__unknown__")
                except json.JSONDecodeError:
                    name = "__unknown__"
                break
        tool_to_examples.setdefault(name or "__unknown__", []).append(ex)

    # Phase 1: Round-robin one example per tool (ensures full exposure)
    import random
    random.seed(42)
    exposure_phase = []
    remaining_by_tool = {}
    for tool_name, exs in tool_to_examples.items():
        random.shuffle(exs)
        exposure_phase.append(exs[0])
        remaining_by_tool[tool_name] = exs[1:]

    random.shuffle(exposure_phase)
    print(f"  Exposure phase: {len(exposure_phase)} examples ({len(tool_to_examples)} unique tools)")

    # Phase 2: Remaining tool examples, shuffled
    remaining_tool = []
    for exs in remaining_by_tool.values():
        remaining_tool.extend(exs)
    random.shuffle(remaining_tool)
    print(f"  Remaining tool examples: {len(remaining_tool)}")

    # Combine: exposure first, then remaining
    ordered_tool = exposure_phase + remaining_tool

    # Interleave non-tool examples evenly throughout
    random.shuffle(non_tool_examples)
    all_ordered = []
    non_tool_idx = 0
    # Insert one non-tool example every N tool examples
    if non_tool_examples:
        interval = max(1, len(ordered_tool) // len(non_tool_examples))
    else:
        interval = len(ordered_tool) + 1

    for i, ex in enumerate(ordered_tool):
        ex["_is_tool"] = True
        all_ordered.append(ex)
        if non_tool_idx < len(non_tool_examples) and (i + 1) % interval == 0:
            non_tool_examples[non_tool_idx]["_is_tool"] = False
            all_ordered.append(non_tool_examples[non_tool_idx])
            non_tool_idx += 1

    # Append any remaining non-tool examples
    while non_tool_idx < len(non_tool_examples):
        non_tool_examples[non_tool_idx]["_is_tool"] = False
        all_ordered.append(non_tool_examples[non_tool_idx])
        non_tool_idx += 1

    print(f"  Final ordered dataset: {len(all_ordered)} examples")

    # Save combined (unsplit) data for the next step
    all_output = args.output_dir / "all_formatted.jsonl"
    with open(all_output, "w", encoding="utf-8") as f:
        for ex in all_ordered:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Also save the schema-included system prompt template for evaluation
    with open(args.output_dir / "system_prompt_with_schema_template.txt", "w", encoding="utf-8") as f:
        f.write(SYSTEM_PROMPT_WITH_SCHEMA)

    print(f"\nFormatted data saved to {all_output}")
    print(f"Total: {len(all_ordered)} examples")


if __name__ == "__main__":
    main()
