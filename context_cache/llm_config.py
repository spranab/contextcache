"""Server-side LLM configuration — keeps API keys out of request bodies.

Instead of passing llm_api_key in every pipeline request, configure LLM
credentials once per tool_id (or as a global default). The pipeline endpoint
resolves config in this order:

    1. Per-tool_id config (from register or admin API)
    2. Global default config (from env or config file)
    3. Request-level override (backward compatible, but discouraged)

Usage:
    # From environment
    store = LLMConfigStore.from_env()  # reads CONTEXTCACHE_LLM_*

    # From JSON file
    store = LLMConfigStore.from_json("llm_config.json")

    # Register per-tool_id
    store.set("merchant", LLMConfig(provider="claude", api_key="sk-..."))

    # Resolve at request time
    config = store.resolve("merchant")  # per-tool_id → default → None
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LLMConfig:
    """LLM configuration for a tool set or global default."""
    provider: str = "claude"          # "claude" or "openai"
    api_key: str = ""                 # API key (never exposed in responses)
    model: Optional[str] = None       # e.g. "claude-sonnet-4-20250514", "gpt-4o"
    base_url: Optional[str] = None    # Custom endpoint (enterprise gateway)
    extra_headers: dict = field(default_factory=dict)  # Additional headers
    system_prompt: Optional[str] = None  # Default system prompt
    timeout: float = 60.0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def to_safe_dict(self) -> dict:
        """Return config dict with API key masked (for admin endpoints)."""
        return {
            "provider": self.provider,
            "api_key": f"...{self.api_key[-4:]}" if len(self.api_key) > 4 else "***",
            "model": self.model,
            "base_url": self.base_url,
            "has_extra_headers": bool(self.extra_headers),
            "system_prompt": self.system_prompt[:50] + "..." if self.system_prompt and len(self.system_prompt) > 50 else self.system_prompt,
            "timeout": self.timeout,
        }


class LLMConfigStore:
    """Manages LLM configurations per tool_id with a global fallback.

    Resolution order: per-tool_id config → global default → None.
    """

    def __init__(self):
        self._configs: dict[str, LLMConfig] = {}
        self._default: Optional[LLMConfig] = None

    def set(self, tool_id: str, config: LLMConfig):
        """Set LLM config for a specific tool_id."""
        self._configs[tool_id] = config

    def set_default(self, config: LLMConfig):
        """Set the global default LLM config."""
        self._default = config

    def get(self, tool_id: str) -> Optional[LLMConfig]:
        """Get config for a specific tool_id (no fallback)."""
        return self._configs.get(tool_id)

    def remove(self, tool_id: str):
        """Remove config for a specific tool_id."""
        self._configs.pop(tool_id, None)

    def resolve(self, tool_id: str) -> Optional[LLMConfig]:
        """Resolve LLM config: per-tool_id → global default → None."""
        config = self._configs.get(tool_id)
        if config and config.is_configured:
            return config
        if self._default and self._default.is_configured:
            return self._default
        return None

    def list_configured(self) -> dict:
        """List all configured tool_ids (safe — no keys exposed)."""
        result = {}
        if self._default:
            result["_default"] = self._default.to_safe_dict()
        for tool_id, config in self._configs.items():
            result[tool_id] = config.to_safe_dict()
        return result

    @classmethod
    def from_env(cls) -> "LLMConfigStore":
        """Load default LLM config from environment variables.

        Reads:
            CONTEXTCACHE_LLM_PROVIDER  (default: "claude")
            CONTEXTCACHE_LLM_API_KEY
            CONTEXTCACHE_LLM_MODEL
            CONTEXTCACHE_LLM_BASE_URL
            CONTEXTCACHE_LLM_SYSTEM_PROMPT
        """
        store = cls()
        api_key = os.environ.get("CONTEXTCACHE_LLM_API_KEY", "")
        if api_key:
            store.set_default(LLMConfig(
                provider=os.environ.get("CONTEXTCACHE_LLM_PROVIDER", "claude"),
                api_key=api_key,
                model=os.environ.get("CONTEXTCACHE_LLM_MODEL"),
                base_url=os.environ.get("CONTEXTCACHE_LLM_BASE_URL"),
                system_prompt=os.environ.get("CONTEXTCACHE_LLM_SYSTEM_PROMPT"),
            ))
        return store

    @classmethod
    def from_json(cls, path: str | Path) -> "LLMConfigStore":
        """Load LLM configs from a JSON file.

        Expected format:
        {
            "default": {
                "provider": "claude",
                "api_key": "sk-...",
                "model": "claude-sonnet-4-20250514"
            },
            "tool_configs": {
                "merchant": {
                    "provider": "openai",
                    "api_key": "sk-...",
                    "model": "gpt-4o",
                    "base_url": "https://gateway.internal.com/v1/chat/completions"
                }
            }
        }
        """
        store = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if "default" in data:
            store.set_default(_config_from_dict(data["default"]))

        for tool_id, cfg in data.get("tool_configs", {}).items():
            store.set(tool_id, _config_from_dict(cfg))

        return store


def _config_from_dict(d: dict) -> LLMConfig:
    return LLMConfig(
        provider=d.get("provider", "claude"),
        api_key=d.get("api_key", ""),
        model=d.get("model"),
        base_url=d.get("base_url"),
        extra_headers=d.get("extra_headers", {}),
        system_prompt=d.get("system_prompt"),
        timeout=d.get("timeout", 60.0),
    )
