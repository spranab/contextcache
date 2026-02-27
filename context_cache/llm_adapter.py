"""LLM adapters for routing tool results to external enterprise LLMs.

Completes the pipeline: route → execute → call enterprise LLM → final answer.
Uses httpx directly (no anthropic/openai SDK needed).

Supports custom endpoints for enterprise gateways (Azure OpenAI, AWS Bedrock,
internal proxies, etc.) via base_url and extra_headers.

Usage:
    # Public API
    adapter = ClaudeAdapter(api_key="sk-...")
    response = await adapter.complete(messages)

    # Enterprise gateway
    adapter = ClaudeAdapter(
        api_key="internal-key",
        base_url="https://llm-gateway.internal.company.com/v1/messages",
        extra_headers={"X-Team-Id": "merchant-analytics", "X-Env": "prod"},
    )
    response = await adapter.complete(messages)

    # Azure OpenAI
    adapter = OpenAIAdapter(
        api_key="azure-key",
        model="gpt-4o",
        base_url="https://myorg.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01",
        extra_headers={"api-key": "azure-key"},
    )
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx


@dataclass
class LLMResponse:
    """Response from an external LLM."""
    content: str
    model: str
    usage: dict
    latency_ms: float


class LLMAdapter(ABC):
    """Base class for external LLM integrations.

    Args:
        api_key: Authentication key for the LLM API.
        model: Model identifier. Uses provider default if None.
        base_url: Custom API endpoint URL. Overrides the provider's default.
                  Use this for enterprise gateways, Azure OpenAI, proxies, etc.
        extra_headers: Additional HTTP headers merged into every request.
                       Useful for gateway auth tokens, team IDs, tracing, etc.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.model = model or self.default_model
        self.base_url = base_url
        self.extra_headers = extra_headers or {}
        self.timeout = timeout

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model to use if none specified."""
        ...

    @property
    @abstractmethod
    def default_url(self) -> str:
        """Default API endpoint URL for this provider."""
        ...

    @property
    def url(self) -> str:
        """Resolved API URL (custom base_url or provider default)."""
        return self.base_url or self.default_url

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """Build provider-specific auth headers. Extra headers are merged on top."""
        ...

    def _merged_headers(self) -> dict[str, str]:
        """Provider headers + extra_headers (extra takes precedence)."""
        headers = self._build_headers()
        headers.update(self.extra_headers)
        return headers

    @abstractmethod
    def format_messages(
        self,
        tool_name: str,
        arguments: dict,
        tool_result: dict | str,
        query: str,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """Format tool result as API-ready messages for this LLM."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send messages to the LLM and return the response."""
        ...


class ClaudeAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API.

    Uses the Messages API with tool_use + tool_result format.
    Default endpoint: https://api.anthropic.com/v1/messages

    For enterprise gateways, set base_url to your gateway's Claude-compatible
    endpoint and pass any required auth via extra_headers.
    """

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-20250514"

    @property
    def default_url(self) -> str:
        return "https://api.anthropic.com/v1/messages"

    def _build_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def format_messages(self, tool_name, arguments, tool_result, query, system_prompt=None):
        tool_use_id = f"toolu_{hashlib.sha256(tool_name.encode()).hexdigest()[:12]}"
        result_content = json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)

        return [
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": tool_name,
                        "input": arguments,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result_content,
                    }
                ],
            },
        ]

    async def complete(self, messages, system_prompt=None, max_tokens=1024, temperature=0.0):
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            body["system"] = system_prompt
        if temperature != 1.0:
            body["temperature"] = temperature

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.url, json=body, headers=self._merged_headers())
            resp.raise_for_status()
            data = resp.json()

        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract text from Claude response
        content_blocks = data.get("content", [])
        text_parts = [b["text"] for b in content_blocks if b.get("type") == "text"]
        content = "\n".join(text_parts)

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            latency_ms=round(latency_ms, 1),
        )


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI-compatible Chat Completions API.

    Works with:
      - OpenAI directly (default)
      - Azure OpenAI (set base_url to your deployment endpoint)
      - Any OpenAI-compatible API (vLLM, Ollama, LiteLLM, etc.)

    For Azure OpenAI, set:
      base_url="https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=2024-02-01"
      extra_headers={"api-key": "your-azure-key"}
    """

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    @property
    def default_url(self) -> str:
        return "https://api.openai.com/v1/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def format_messages(self, tool_name, arguments, tool_result, query, system_prompt=None):
        call_id = f"call_{hashlib.sha256(tool_name.encode()).hexdigest()[:12]}"
        result_content = json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend([
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_content,
            },
        ])
        return messages

    async def complete(self, messages, system_prompt=None, max_tokens=1024, temperature=0.0):
        # If system_prompt provided and not already in messages, prepend it
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}] + messages

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.url, json=body, headers=self._merged_headers())
            resp.raise_for_status()
            data = resp.json()

        latency_ms = (time.perf_counter() - t0) * 1000

        choices = data.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""

        return LLMResponse(
            content=content or "",
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            latency_ms=round(latency_ms, 1),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LLM_ADAPTER_REGISTRY: dict[str, type[LLMAdapter]] = {
    "claude": ClaudeAdapter,
    "openai": OpenAIAdapter,
}


def get_llm_adapter(
    provider: str,
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> LLMAdapter:
    """Get an LLM adapter by provider name.

    Args:
        provider: "claude" or "openai".
        api_key: API key for the provider.
        model: Optional model override (uses provider default if None).
        base_url: Custom endpoint URL (enterprise gateway, Azure, etc.).
        extra_headers: Additional HTTP headers for every request.
        timeout: Request timeout in seconds.

    Returns:
        Configured LLMAdapter instance.
    """
    adapter_cls = LLM_ADAPTER_REGISTRY.get(provider)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available: {list(LLM_ADAPTER_REGISTRY.keys())}"
        )
    return adapter_cls(
        api_key=api_key,
        model=model,
        base_url=base_url,
        extra_headers=extra_headers,
        timeout=timeout,
    )
