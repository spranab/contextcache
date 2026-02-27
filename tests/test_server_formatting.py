"""Tests for server-side formatting functions (_format_for_claude, _format_for_openai, _mock_tool_result)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "serve"))
from serve_context_cache import _format_for_claude, _format_for_openai, _mock_tool_result


class TestFormatForClaude:
    def test_structure(self):
        messages = _format_for_claude("get_weather", {"location": "NYC"}, {"temp": 72}, "Weather?")
        assert len(messages) == 3

    def test_user_role_first(self):
        messages = _format_for_claude("get_weather", {}, {}, "Weather?")
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Weather?"

    def test_assistant_tool_use(self):
        messages = _format_for_claude("get_weather", {"location": "NYC"}, {}, "q")
        assistant = messages[1]
        assert assistant["role"] == "assistant"
        blocks = assistant["content"]
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["name"] == "get_weather"
        assert blocks[0]["input"] == {"location": "NYC"}

    def test_tool_result_message(self):
        result = {"temperature": 72}
        messages = _format_for_claude("get_weather", {}, result, "q")
        tool_result = messages[2]
        assert tool_result["role"] == "user"
        block = tool_result["content"][0]
        assert block["type"] == "tool_result"
        assert json.loads(block["content"]) == result

    def test_ids_match(self):
        messages = _format_for_claude("get_weather", {}, {}, "q")
        use_id = messages[1]["content"][0]["id"]
        result_id = messages[2]["content"][0]["tool_use_id"]
        assert use_id == result_id
        assert use_id.startswith("toolu_")


class TestFormatForOpenAI:
    def test_structure(self):
        messages = _format_for_openai("get_weather", {"location": "NYC"}, {"temp": 72}, "Weather?")
        assert len(messages) == 3

    def test_user_role_first(self):
        messages = _format_for_openai("get_weather", {}, {}, "Weather?")
        assert messages[0]["role"] == "user"

    def test_assistant_tool_calls(self):
        messages = _format_for_openai("get_weather", {"location": "NYC"}, {}, "q")
        assistant = messages[1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] is None
        tc = assistant["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"location": "NYC"}

    def test_tool_message(self):
        result = {"temperature": 72}
        messages = _format_for_openai("get_weather", {}, result, "q")
        tool_msg = messages[2]
        assert tool_msg["role"] == "tool"
        assert json.loads(tool_msg["content"]) == result

    def test_ids_match(self):
        messages = _format_for_openai("get_weather", {}, {}, "q")
        call_id = messages[1]["tool_calls"][0]["id"]
        assert messages[2]["tool_call_id"] == call_id
        assert call_id.startswith("call_")


class TestMockToolResult:
    def test_returns_dict(self):
        result = _mock_tool_result("get_weather", {"location": "NYC"})
        assert isinstance(result, dict)

    def test_contains_tool_name(self):
        result = _mock_tool_result("get_weather", {"location": "NYC"})
        assert result["tool"] == "get_weather"

    def test_contains_status(self):
        result = _mock_tool_result("ping", {})
        assert "status" in result

    def test_contains_arguments_received(self):
        args = {"location": "NYC", "units": "celsius"}
        result = _mock_tool_result("get_weather", args)
        assert result["data"]["arguments_received"] == args
