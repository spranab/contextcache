"""Tests for LLMConfigStore — server-side LLM credential management."""

import json
import os
import tempfile

import pytest

from context_cache.llm_config import LLMConfig, LLMConfigStore


class TestLLMConfig:
    def test_is_configured(self):
        config = LLMConfig(api_key="sk-test")
        assert config.is_configured is True

    def test_not_configured_empty_key(self):
        config = LLMConfig(api_key="")
        assert config.is_configured is False

    def test_to_safe_dict_masks_key(self):
        config = LLMConfig(api_key="sk-ant-very-secret-key")
        safe = config.to_safe_dict()
        assert safe["api_key"] == "...-key"
        assert "secret" not in safe["api_key"]

    def test_to_safe_dict_short_key(self):
        config = LLMConfig(api_key="abc")
        safe = config.to_safe_dict()
        assert safe["api_key"] == "***"

    def test_to_safe_dict_truncates_system_prompt(self):
        config = LLMConfig(api_key="k", system_prompt="x" * 100)
        safe = config.to_safe_dict()
        assert safe["system_prompt"].endswith("...")
        assert len(safe["system_prompt"]) < 100

    def test_defaults(self):
        config = LLMConfig()
        assert config.provider == "claude"
        assert config.model is None
        assert config.timeout == 60.0


class TestLLMConfigStore:
    def test_set_and_get(self):
        store = LLMConfigStore()
        store.set("merchant", LLMConfig(api_key="sk-1"))
        config = store.get("merchant")
        assert config is not None
        assert config.api_key == "sk-1"

    def test_get_missing(self):
        store = LLMConfigStore()
        assert store.get("nonexistent") is None

    def test_set_default(self):
        store = LLMConfigStore()
        store.set_default(LLMConfig(api_key="sk-default"))
        assert store._default.api_key == "sk-default"

    def test_resolve_per_tool_id(self):
        store = LLMConfigStore()
        store.set_default(LLMConfig(api_key="sk-default"))
        store.set("merchant", LLMConfig(api_key="sk-merchant"))
        config = store.resolve("merchant")
        assert config.api_key == "sk-merchant"

    def test_resolve_falls_back_to_default(self):
        store = LLMConfigStore()
        store.set_default(LLMConfig(api_key="sk-default"))
        config = store.resolve("unknown_tool")
        assert config.api_key == "sk-default"

    def test_resolve_returns_none_when_empty(self):
        store = LLMConfigStore()
        assert store.resolve("anything") is None

    def test_resolve_skips_unconfigured_tool(self):
        """If tool_id config has empty api_key, fall back to default."""
        store = LLMConfigStore()
        store.set_default(LLMConfig(api_key="sk-default"))
        store.set("merchant", LLMConfig(api_key=""))  # Not configured
        config = store.resolve("merchant")
        assert config.api_key == "sk-default"

    def test_remove(self):
        store = LLMConfigStore()
        store.set("merchant", LLMConfig(api_key="sk-1"))
        store.remove("merchant")
        assert store.get("merchant") is None

    def test_remove_nonexistent(self):
        store = LLMConfigStore()
        store.remove("nonexistent")  # Should not raise

    def test_list_configured(self):
        store = LLMConfigStore()
        store.set_default(LLMConfig(api_key="sk-default"))
        store.set("merchant", LLMConfig(api_key="sk-merchant", provider="openai"))
        result = store.list_configured()
        assert "_default" in result
        assert "merchant" in result
        assert result["merchant"]["provider"] == "openai"
        # Keys are masked
        assert "sk-merchant" not in str(result)

    def test_from_json(self):
        data = {
            "default": {
                "provider": "claude",
                "api_key": "sk-default-key",
                "model": "claude-sonnet-4-20250514",
            },
            "tool_configs": {
                "merchant": {
                    "provider": "openai",
                    "api_key": "sk-openai-key",
                    "model": "gpt-4o",
                    "base_url": "https://gateway.internal.com",
                },
                "support": {
                    "provider": "claude",
                    "api_key": "sk-support-key",
                },
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            store = LLMConfigStore.from_json(f.name)

        assert store._default.api_key == "sk-default-key"
        assert store.get("merchant").provider == "openai"
        assert store.get("merchant").base_url == "https://gateway.internal.com"
        assert store.get("support").api_key == "sk-support-key"

    def test_from_env(self):
        env = {
            "CONTEXTCACHE_LLM_API_KEY": "sk-from-env",
            "CONTEXTCACHE_LLM_PROVIDER": "openai",
            "CONTEXTCACHE_LLM_MODEL": "gpt-4o",
        }
        old_env = {}
        for k, v in env.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            store = LLMConfigStore.from_env()
            assert store._default is not None
            assert store._default.api_key == "sk-from-env"
            assert store._default.provider == "openai"
            assert store._default.model == "gpt-4o"
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_from_env_no_key(self):
        """If CONTEXTCACHE_LLM_API_KEY not set, no default is created."""
        old = os.environ.pop("CONTEXTCACHE_LLM_API_KEY", None)
        try:
            store = LLMConfigStore.from_env()
            assert store._default is None
        finally:
            if old:
                os.environ["CONTEXTCACHE_LLM_API_KEY"] = old
