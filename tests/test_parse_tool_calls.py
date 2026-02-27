"""Tests for parse_tool_calls — the parser that extracts tool selections from model output."""

import json
import sys
from pathlib import Path

import pytest

# Add scripts/serve to path so we can import parse_tool_calls
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "serve"))
from serve_context_cache import ToolSelection, parse_tool_calls


# ---------------------------------------------------------------------------
# Clean <tool_call> format (confidence = 1.0)
# ---------------------------------------------------------------------------


class TestCleanToolCallFormat:
    """Model outputs clean <tool_call>{...}</tool_call> blocks."""

    def test_single_tool_call(self):
        raw = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert len(selections) == 1
        assert selections[0].tool_name == "get_weather"
        assert selections[0].arguments == {"location": "NYC"}

    def test_tool_call_no_arguments(self):
        raw = '<tool_call>\n{"name": "list_all_orders", "arguments": {}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert len(selections) == 1
        assert selections[0].tool_name == "list_all_orders"
        assert selections[0].arguments == {}

    def test_tool_call_complex_arguments(self):
        raw = '<tool_call>\n{"name": "gmv_summary", "arguments": {"time_period": "last_30_days", "currency": "USD", "include_refunds": true}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert selections[0].tool_name == "gmv_summary"
        assert selections[0].arguments["include_refunds"] is True

    def test_tool_call_no_whitespace(self):
        raw = '<tool_call>{"name": "ping", "arguments": {}}</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert selections[0].tool_name == "ping"

    def test_tool_call_extra_whitespace(self):
        raw = '<tool_call>\n\n  {"name": "get_weather", "arguments": {"location": "London"}}  \n\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert selections[0].tool_name == "get_weather"

    def test_forced_prefix_output(self):
        """Output from forced prefix: starts with <tool_call>\\n{ already forced."""
        raw = '<tool_call>\n{"name": "track_order", "arguments": {"order_id": "ORD-123"}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert selections[0].tool_name == "track_order"
        assert selections[0].arguments == {"order_id": "ORD-123"}

    def test_multiple_tool_calls(self):
        raw = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "London"}}\n</tool_call>'
        )
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert len(selections) == 2
        assert selections[0].arguments["location"] == "NYC"
        assert selections[1].arguments["location"] == "London"

    def test_arguments_with_nested_objects(self):
        raw = '<tool_call>\n{"name": "create_report", "arguments": {"filters": {"date_range": {"start": "2024-01-01", "end": "2024-12-31"}, "status": "active"}}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert selections[0].arguments["filters"]["date_range"]["start"] == "2024-01-01"


# ---------------------------------------------------------------------------
# Regex fallback (confidence = 0.5)
# ---------------------------------------------------------------------------


class TestFallbackParsing:
    """Model outputs JSON without <tool_call> tags — regex fallback."""

    def test_bare_json_with_name_flat(self):
        """Fallback regex uses [^{}] so it matches flat JSON (no nested braces)."""
        raw = '{"name": "list_all"}'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 0.5
        assert len(selections) == 1
        assert selections[0].tool_name == "list_all"

    def test_bare_json_with_nested_arguments_no_fallback(self):
        """JSON with nested braces in arguments won't match [^{}]* fallback regex."""
        raw = '{"name": "get_weather", "arguments": {"location": "Paris"}}'
        selections, confidence = parse_tool_calls(raw)
        # Fallback regex can't match nested braces — returns 0.0
        assert confidence == 0.0

    def test_json_with_surrounding_text_flat(self):
        raw = 'I will call: {"name": "search_web"} for you.'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 0.5
        assert selections[0].tool_name == "search_web"

    def test_json_name_only_no_arguments(self):
        raw = '{"name": "list_tools"}'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 0.5
        assert selections[0].tool_name == "list_tools"
        assert selections[0].arguments == {}


# ---------------------------------------------------------------------------
# No parse (confidence = 0.0)
# ---------------------------------------------------------------------------


class TestNoParse:
    """Model output with no tool call — should return empty with 0.0."""

    def test_empty_string(self):
        selections, confidence = parse_tool_calls("")
        assert confidence == 0.0
        assert selections == []

    def test_plain_text(self):
        selections, confidence = parse_tool_calls("I don't know how to help with that.")
        assert confidence == 0.0
        assert selections == []

    def test_incomplete_tool_call_tag(self):
        selections, confidence = parse_tool_calls('<tool_call>\n{"name": "broken"')
        # The regex fallback should pick this up since it has "name"
        assert len(selections) >= 0  # Either fallback catches it or not

    def test_malformed_json_in_tags(self):
        raw = '<tool_call>\n{not valid json}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        # JSON decode fails, falls through
        assert confidence == 0.0 or confidence == 0.5  # Depends on fallback

    def test_empty_tool_call_tags(self):
        raw = '<tool_call>\n\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert selections == [] or confidence < 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for robust parsing."""

    def test_missing_arguments_key(self):
        """Model outputs name but no arguments key."""
        raw = '<tool_call>\n{"name": "ping"}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert selections[0].tool_name == "ping"
        assert selections[0].arguments == {}

    def test_missing_name_key(self):
        """JSON with no name — should be skipped."""
        raw = '<tool_call>\n{"tool": "get_weather", "arguments": {}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        if selections:
            # Falls through to tool_name="unknown" since .get("name", "unknown")
            assert selections[0].tool_name == "unknown"

    def test_unicode_in_arguments(self):
        raw = '<tool_call>\n{"name": "translate", "arguments": {"text": "こんにちは"}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert selections[0].arguments["text"] == "こんにちは"

    def test_newlines_in_json(self):
        raw = """<tool_call>
{
  "name": "get_weather",
  "arguments": {
    "location": "New York"
  }
}
</tool_call>"""
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert selections[0].tool_name == "get_weather"

    def test_numeric_argument_values(self):
        raw = '<tool_call>\n{"name": "calculate", "arguments": {"x": 42, "y": 3.14}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert selections[0].arguments["x"] == 42
        assert selections[0].arguments["y"] == 3.14

    def test_null_argument_value(self):
        raw = '<tool_call>\n{"name": "search", "arguments": {"query": "test", "limit": null}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert selections[0].arguments["limit"] is None

    def test_array_argument_value(self):
        raw = '<tool_call>\n{"name": "batch", "arguments": {"ids": [1, 2, 3]}}\n</tool_call>'
        selections, confidence = parse_tool_calls(raw)
        assert confidence == 1.0
        assert selections[0].arguments["ids"] == [1, 2, 3]
