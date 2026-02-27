"""LLM adapters for routing tool results to external enterprise LLMs.

Completes the pipeline: route -> execute -> call enterprise LLM -> final answer.
Uses httpx directly (no anthropic/openai SDK needed).

Supports custom endpoints for enterprise gateways (Azure OpenAI, AWS Bedrock,
internal proxies, etc.) via base_url and extra_headers.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx

_MAX_RETRIES = 3
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_INITIAL_BACKOFF_S = 1.0


@dataclass
class LLMResponse:
    """Response from an external LLM."""
    content: str
    model: str
    usage: dict
    latency_ms: float


def _unique_tool_id(prefix: str, tool_name: str) -> str:
    """Generate a unique tool call ID using UUID to avoid collisions."""
    short_uuid = uuid.uuid4().hex[:12]
    return f"{prefix}_{short_uuid}"


class LLMAdapter(ABC):
    """Base class for external LLM integrations.

    Args:
        api_key: Authentication key for the LLM API.
        model: Model identifier. Uses provider default if None.
        base_url: Custom API endpoint URL for enterprise gateways, Azure, etc.
        extra_headers: Additional HTTP headers merged into every request.
        timeout: Request timeout in seconds.
        max_retries: Max retry attempts for transient errors (429, 5xx).
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        max_retries: int = _MAX_RETRIES,
    ):
        self.api_key = api_key
        self.model = model or self.default_model
        self.base_url = base_url
        self.extra_headers = extra_headers or {}
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    @abstractmethod
    def default_model(self) -> str: ...

    @property
    @abstractmethod
    def default_url(self) -> str: ...

    @property
    def url(self) -> str:
        return self.base_url or self.default_url

    @abstractmethod
    def _build_headers(self) -> dict[str, str]: ...

    def _merged_headers(self) -> dict[str, str]:
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
    ) -> list[dict]: ...

    async def _request_with_retry(self, body: dict) -> dict:
        """Send request with exponential backoff retry on transient errors."""
        import asyncio

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        self.url, json=body, headers=self._merged_headers()
                    )
                    if resp.status_code in _RETRY_STATUS_CODES and attempt < self.max_retries:
                        retry_after = resp.headers.get("retry-after")
                        wait = min(float(retry_after), 30.0) if retry_after else _INITIAL_BACKOFF_S * (2 ** attempt)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TimeoutException:
                last_error = httpx.TimeoutException(
                    f"Request timed out after {self.timeout}s (attempt {attempt + 1}/{self.max_retries + 1})"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(_INITIAL_BACKOFF_S * (2 ** attempt))
            except httpx.HTTPStatusError as e:
                if e.response.status_code in _RETRY_STATUS_CODES and attempt < self.max_retries:
                    await asyncio.sleep(_INITIAL_BACKOFF_S * (2 ** attempt))
                    last_error = e
                else:
                    raise
        raise last_error or RuntimeError("Request failed after retries")

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class ClaudeAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API (Messages API with tool_use)."""

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
        tool_use_id = _unique_tool_id("toolu", tool_name)
        result_content = json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)
        return [
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use", "id": tool_use_id,
                    "name": tool_name, "input": arguments,
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                }],
            },
        ]

    async def complete(self, messages, system_prompt=None, max_tokens=1024, temperature=0.0):
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system_prompt:
            body["system"] = system_prompt

        t0 = time.perf_counter()
        data = await self._request_with_retry(body)
        latency_ms = (time.perf_counter() - t0) * 1000

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

    Works with OpenAI, Azure OpenAI, vLLM, Ollama, LiteLLM, etc.
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
        call_id = _unique_tool_id("call", tool_name)
        result_content = json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend([
            {"role": "user", "content": query},
            {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(arguments)},
                }],
            },
            {"role": "tool", "tool_call_id": call_id, "content": result_content},
        ])
        return messages

    async def complete(self, messages, system_prompt=None, max_tokens=1024, temperature=0.0):
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}] + messages

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        t0 = time.perf_counter()
        data = await self._request_with_retry(body)
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
    """Get an LLM adapter by provider name."""
    adapter_cls = LLM_ADAPTER_REGISTRY.get(provider)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available: {list(LLM_ADAPTER_REGISTRY.keys())}"
        )
    return adapter_cls(
        api_key=api_key, model=model, base_url=base_url,
        extra_headers=extra_headers, timeout=timeout,
    )
