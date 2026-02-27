"""Tests for LLM adapters — message formatting and adapter registry."""

import hashlib
import json

import pytest

from context_cache.llm_adapter import (
    ClaudeAdapter,
    LLM_ADAPTER_REGISTRY,
    OpenAIAdapter,
    get_llm_adapter,
)


# ---------------------------------------------------------------------------
# ClaudeAdapter message formatting
# ---------------------------------------------------------------------------


class TestClaudeAdapterFormatting:
    """ClaudeAdapter.format_messages produces valid Claude API messages."""

    def setup_method(self):
        self.adapter = ClaudeAdapter(api_key="test-key")

    def test_message_structure(self):
        messages = self.adapter.format_messages(
            tool_name="get_weather",
            arguments={"location": "NYC"},
            tool_result={"temperature": 72, "condition": "sunny"},
            query="What's the weather?",
        )
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_user_message_contains_query(self):
        messages = self.adapter.format_messages(
            "get_weather", {"location": "NYC"}, {"temp": 72}, "What's the weather?"
        )
        assert messages[0]["content"] == "What's the weather?"

    def test_assistant_message_has_tool_use(self):
        messages = self.adapter.format_messages(
            "get_weather", {"location": "NYC"}, {"temp": 72}, "Weather?"
        )
        content_blocks = messages[1]["content"]
        assert isinstance(content_blocks, list)
        assert len(content_blocks) == 1
        block = content_blocks[0]
        assert block["type"] == "tool_use"
        assert block["name"] == "get_weather"
        assert block["input"] == {"location": "NYC"}
        assert block["id"].startswith("toolu_")

    def test_tool_result_message(self):
        result = {"temperature": 72}
        messages = self.adapter.format_messages(
            "get_weather", {}, result, "Weather?"
        )
        content_blocks = messages[2]["content"]
        assert isinstance(content_blocks, list)
        block = content_blocks[0]
        assert block["type"] == "tool_result"
        assert json.loads(block["content"]) == result

    def test_tool_use_id_matches_tool_result_id(self):
        messages = self.adapter.format_messages(
            "get_weather", {}, {}, "Weather?"
        )
        tool_use_id = messages[1]["content"][0]["id"]
        tool_result_id = messages[2]["content"][0]["tool_use_id"]
        assert tool_use_id == tool_result_id

    def test_tool_use_id_is_unique_per_call(self):
        """Each call produces a unique ID (UUID-based to avoid collisions)."""
        m1 = self.adapter.format_messages("get_weather", {}, {}, "q1")
        m2 = self.adapter.format_messages("get_weather", {}, {}, "q2")
        assert m1[1]["content"][0]["id"] != m2[1]["content"][0]["id"]
        assert m1[1]["content"][0]["id"].startswith("toolu_")
        assert m2[1]["content"][0]["id"].startswith("toolu_")

    def test_tool_use_id_format(self):
        m1 = self.adapter.format_messages("get_weather", {}, {}, "q")
        tool_id = m1[1]["content"][0]["id"]
        assert tool_id.startswith("toolu_")
        assert len(tool_id) == len("toolu_") + 12  # prefix + 12 hex chars

    def test_string_tool_result(self):
        """tool_result can be a string (not just dict)."""
        messages = self.adapter.format_messages(
            "ping", {}, "pong", "Ping?"
        )
        assert messages[2]["content"][0]["content"] == "pong"

    def test_dict_tool_result_is_json_serialized(self):
        result = {"data": [1, 2, 3]}
        messages = self.adapter.format_messages(
            "get_data", {}, result, "Get data"
        )
        content = messages[2]["content"][0]["content"]
        assert json.loads(content) == result


# ---------------------------------------------------------------------------
# OpenAIAdapter message formatting
# ---------------------------------------------------------------------------


class TestOpenAIAdapterFormatting:
    """OpenAIAdapter.format_messages produces valid OpenAI API messages."""

    def setup_method(self):
        self.adapter = OpenAIAdapter(api_key="test-key")

    def test_message_structure_without_system(self):
        messages = self.adapter.format_messages(
            "get_weather", {"location": "NYC"}, {"temp": 72}, "Weather?"
        )
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "tool"

    def test_message_structure_with_system_prompt(self):
        messages = self.adapter.format_messages(
            "get_weather", {}, {}, "Weather?", system_prompt="You are a weather bot."
        )
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a weather bot."

    def test_assistant_has_tool_calls(self):
        messages = self.adapter.format_messages(
            "get_weather", {"location": "NYC"}, {}, "Weather?"
        )
        assistant = messages[1]
        assert assistant["content"] is None
        assert len(assistant["tool_calls"]) == 1
        tc = assistant["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"location": "NYC"}

    def test_tool_message_has_matching_id(self):
        messages = self.adapter.format_messages(
            "get_weather", {}, {}, "Weather?"
        )
        call_id = messages[1]["tool_calls"][0]["id"]
        assert messages[2]["tool_call_id"] == call_id

    def test_tool_result_json_serialized(self):
        result = {"temperature": 72, "unit": "F"}
        messages = self.adapter.format_messages(
            "get_weather", {}, result, "Weather?"
        )
        assert json.loads(messages[2]["content"]) == result

    def test_call_id_starts_with_call_prefix(self):
        messages = self.adapter.format_messages("ping", {}, {}, "q")
        assert messages[1]["tool_calls"][0]["id"].startswith("call_")


# ---------------------------------------------------------------------------
# Adapter configuration
# ---------------------------------------------------------------------------


class TestAdapterConfig:
    """Adapter initialization and configuration."""

    def test_claude_default_model(self):
        adapter = ClaudeAdapter(api_key="k")
        assert "claude" in adapter.model

    def test_openai_default_model(self):
        adapter = OpenAIAdapter(api_key="k")
        assert adapter.model == "gpt-4o"

    def test_custom_model(self):
        adapter = ClaudeAdapter(api_key="k", model="claude-haiku-4-5-20251001")
        assert adapter.model == "claude-haiku-4-5-20251001"

    def test_claude_default_url(self):
        adapter = ClaudeAdapter(api_key="k")
        assert adapter.url == "https://api.anthropic.com/v1/messages"

    def test_openai_default_url(self):
        adapter = OpenAIAdapter(api_key="k")
        assert adapter.url == "https://api.openai.com/v1/chat/completions"

    def test_custom_base_url(self):
        adapter = ClaudeAdapter(api_key="k", base_url="https://gateway.internal.com/v1/messages")
        assert adapter.url == "https://gateway.internal.com/v1/messages"

    def test_extra_headers_merged(self):
        adapter = ClaudeAdapter(
            api_key="k",
            extra_headers={"X-Team-Id": "analytics", "X-Env": "prod"},
        )
        headers = adapter._merged_headers()
        assert headers["x-api-key"] == "k"
        assert headers["X-Team-Id"] == "analytics"
        assert headers["X-Env"] == "prod"

    def test_extra_headers_override_provider(self):
        """Extra headers take precedence over provider defaults."""
        adapter = ClaudeAdapter(
            api_key="k",
            extra_headers={"x-api-key": "override-key"},
        )
        headers = adapter._merged_headers()
        assert headers["x-api-key"] == "override-key"

    def test_openai_auth_header(self):
        adapter = OpenAIAdapter(api_key="sk-test")
        headers = adapter._build_headers()
        assert headers["Authorization"] == "Bearer sk-test"


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    """get_llm_adapter() factory function."""

    def test_get_claude_adapter(self):
        adapter = get_llm_adapter("claude", api_key="k")
        assert isinstance(adapter, ClaudeAdapter)

    def test_get_openai_adapter(self):
        adapter = get_llm_adapter("openai", api_key="k")
        assert isinstance(adapter, OpenAIAdapter)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_adapter("gemini", api_key="k")

    def test_factory_passes_all_params(self):
        adapter = get_llm_adapter(
            "claude",
            api_key="my-key",
            model="claude-haiku-4-5-20251001",
            base_url="https://gateway.com/v1/messages",
            extra_headers={"X-Team": "test"},
            timeout=120.0,
        )
        assert adapter.api_key == "my-key"
        assert adapter.model == "claude-haiku-4-5-20251001"
        assert adapter.url == "https://gateway.com/v1/messages"
        assert adapter.extra_headers == {"X-Team": "test"}
        assert adapter.timeout == 120.0

    def test_registry_contents(self):
        assert "claude" in LLM_ADAPTER_REGISTRY
        assert "openai" in LLM_ADAPTER_REGISTRY
        assert len(LLM_ADAPTER_REGISTRY) == 2
