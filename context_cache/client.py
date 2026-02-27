"""ContextCache Python SDK — lightweight client for the routing API.

Usage:
    from contextcache import ContextCacheClient

    client = ContextCacheClient("http://localhost:8421", api_key="your-key")
    client.register_tools("merchant", tools=[...])

    # Route only — get the tool selection
    result = client.route("merchant", "What's my GMV?")
    print(result.tool_name)       # "gmv_summary"
    print(result.arguments)       # {"time_period": "last_30_days"}
    print(result.confidence)      # 1.0

    # Format only — get LLM-ready messages
    pipeline = client.pipeline("merchant", "What's my GMV?", llm_format="claude")
    print(pipeline.llm_context)   # Ready for Anthropic SDK

    # Full end-to-end — route + execute + call Claude for final answer
    pipeline = client.pipeline(
        "merchant", "What's my GMV?",
        llm_format="claude",
        llm_api_key="sk-ant-...",
        llm_model="claude-sonnet-4-20250514",
    )
    print(pipeline.llm_response)  # Claude's final answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 3


@dataclass
class RouteResult:
    """Result from tool routing."""
    tool_name: str
    arguments: dict
    confidence: float
    raw_response: str
    latency_ms: float
    cache_hit: bool
    timings: dict


@dataclass
class PipelineResult:
    """Result from the full pipeline (route + execute + format + optional LLM call)."""
    tool_name: str
    arguments: dict
    tool_result: Any
    llm_context: dict | list
    llm_response: str | None
    confidence: float
    latency_ms: float
    timings: dict


@dataclass
class RegisterResult:
    """Result from tool registration."""
    tool_id: str
    num_tools: int
    cache_hash: str
    compile_ms: float
    cache_size_mb: float
    status: str


class ContextCacheClient:
    """Python client for the ContextCache API.

    Args:
        base_url: Server URL (e.g., "http://localhost:8421").
        api_key: Optional API key for authenticated endpoints.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8421",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        # Retry with exponential backoff on transient errors
        retry = Retry(
            total=_MAX_RETRIES,
            backoff_factor=1.0,
            status_forcelist=_RETRY_STATUS,
            allowed_methods=["GET", "POST", "DELETE"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        if api_key:
            self._session.headers["X-API-Key"] = api_key

    def health(self) -> dict:
        """Check server health."""
        r = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def register_tools(self, tool_id: str, tools: list[dict]) -> RegisterResult:
        """Register a set of tools under a namespace.

        Args:
            tool_id: Unique identifier for this tool set.
            tools: List of tool schemas in OpenAI function-calling format.

        Returns:
            RegisterResult with compilation details.
        """
        r = self._session.post(
            f"{self.base_url}/v2/tools",
            json={"tool_id": tool_id, "tools": tools},
            timeout=max(self.timeout, 120),  # Compilation can take time
        )
        r.raise_for_status()
        data = r.json()
        return RegisterResult(
            tool_id=data["tool_id"],
            num_tools=data["num_tools"],
            cache_hash=data["cache_hash"],
            compile_ms=data["compile_ms"],
            cache_size_mb=data["cache_size_mb"],
            status=data["status"],
        )

    def route(
        self,
        tool_id: str,
        query: str,
        max_new_tokens: int = 256,
    ) -> RouteResult:
        """Route a query to the best matching tool.

        Args:
            tool_id: Which tool set to route against.
            query: User's natural language query.
            max_new_tokens: Max tokens for model generation.

        Returns:
            RouteResult with selected tool, arguments, and confidence.
        """
        r = self._session.post(
            f"{self.base_url}/route",
            json={"tool_id": tool_id, "query": query, "max_new_tokens": max_new_tokens},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        selections = data.get("selections", [])
        if not selections:
            import logging
            logging.getLogger("contextcache").warning(
                "No tool selections returned for query: %s", query[:100]
            )
        first = selections[0] if selections else {"tool_name": "unknown", "arguments": {}}
        return RouteResult(
            tool_name=first["tool_name"],
            arguments=first["arguments"],
            confidence=data.get("confidence", 0.0),
            raw_response=data.get("raw_response", ""),
            latency_ms=data.get("timings", {}).get("total_ms", 0),
            cache_hit=data.get("timings", {}).get("cache_hit", False),
            timings=data.get("timings", {}),
        )

    def pipeline(
        self,
        tool_id: str,
        query: str,
        llm_format: str = "claude",
        tool_executor: str = "mock",
        max_new_tokens: int = 256,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        llm_system_prompt: str | None = None,
        llm_base_url: str | None = None,
        llm_headers: dict[str, str] | None = None,
    ) -> PipelineResult:
        """Full pipeline: route -> execute -> format -> optionally call enterprise LLM.

        Args:
            tool_id: Which tool set to route against.
            query: User's natural language query.
            llm_format: LLM provider/format — "claude", "openai", or "raw".
            tool_executor: "mock" for simulated results, or a URL for real execution.
            max_new_tokens: Max tokens for model generation.
            llm_api_key: If provided, calls the enterprise LLM and returns its response.
            llm_model: LLM model to use (e.g. "claude-sonnet-4-20250514", "gpt-4o").
            llm_system_prompt: Optional system prompt for the enterprise LLM.
            llm_base_url: Custom LLM endpoint (enterprise gateway, Azure OpenAI, etc.).
            llm_headers: Extra HTTP headers for the LLM request.

        Returns:
            PipelineResult with tool result, LLM-ready context, and optionally
            the LLM's final response in llm_response.
        """
        body = {
            "tool_id": tool_id,
            "query": query,
            "llm_format": llm_format,
            "tool_executor": tool_executor,
            "max_new_tokens": max_new_tokens,
        }
        if llm_api_key:
            body["llm_api_key"] = llm_api_key
        if llm_model:
            body["llm_model"] = llm_model
        if llm_system_prompt:
            body["llm_system_prompt"] = llm_system_prompt
        if llm_base_url:
            body["llm_base_url"] = llm_base_url
        if llm_headers:
            body["llm_headers"] = llm_headers

        # LLM calls can take longer
        timeout = max(self.timeout, 60) if llm_api_key else self.timeout

        r = self._session.post(
            f"{self.base_url}/v2/pipeline", json=body, timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        return PipelineResult(
            tool_name=data.get("selected_tool", "unknown"),
            arguments=data.get("arguments", {}),
            tool_result=data.get("tool_result"),
            llm_context=data.get("llm_context", {}),
            llm_response=data.get("llm_response"),
            confidence=data.get("confidence", 0.0),
            latency_ms=data.get("timings", {}).get("total_ms", 0),
            timings=data.get("timings", {}),
        )

    def list_tool_sets(self) -> list[dict]:
        """List all registered tool sets."""
        r = self._session.get(f"{self.base_url}/v2/registry", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("tool_sets", [])

    def remove_tools(self, tool_id: str) -> dict:
        """Remove a tool set."""
        r = self._session.delete(
            f"{self.base_url}/v2/tools/{tool_id}", timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    # --- Server-side LLM config ---

    def configure_llm(
        self,
        tool_id: str,
        provider: str = "claude",
        api_key: str = "",
        model: str | None = None,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
        system_prompt: str | None = None,
    ) -> dict:
        """Configure server-side LLM credentials for a tool_id.

        Use tool_id="_default" to set the global default.
        Once configured, pipeline() calls will use these credentials
        automatically — no need to pass llm_api_key per request.

        Args:
            tool_id: Tool set ID (or "_default" for global).
            provider: "claude" or "openai".
            api_key: API key for the LLM provider.
            model: Model name override.
            base_url: Custom endpoint URL.
            extra_headers: Additional HTTP headers.
            system_prompt: Default system prompt.

        Returns:
            Config status with masked credentials.
        """
        body = {"provider": provider, "api_key": api_key}
        if model:
            body["model"] = model
        if base_url:
            body["base_url"] = base_url
        if extra_headers:
            body["extra_headers"] = extra_headers
        if system_prompt:
            body["system_prompt"] = system_prompt

        r = self._session.post(
            f"{self.base_url}/admin/llm-config/{tool_id}",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def list_llm_configs(self) -> dict:
        """List all server-side LLM configs (API keys masked)."""
        r = self._session.get(
            f"{self.base_url}/admin/llm-config", timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def remove_llm_config(self, tool_id: str) -> dict:
        """Remove server-side LLM config for a tool_id."""
        r = self._session.delete(
            f"{self.base_url}/admin/llm-config/{tool_id}", timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
