"""Confidence-gated route-first orchestrator.

Coordinates: ToolRouter (local model) -> ConfidenceGate -> Execute -> LLM synthesis.

The local model handles BOTH tool routing AND parameter extraction in one shot
using cached KV state. The external LLM never sees tool schemas — it only
synthesizes a user-friendly response from the tool result.

Pipeline:
  1. ToolRouter.route() -> tool_name + arguments + confidence  (local, cached ~550ms)
  2. Confidence gate:
     - HIGH/LOW: Execute tool with extracted args
     - NO_TOOL: LLM gets raw query -> conversational response
  3. Execute tool with params
  4. LLM synthesizes final answer from query + tool result (no schema needed)
  5. If LLM requests another tool call, loop back to step 1-4
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from context_cache.llm_adapter import LLMAdapter, LLMResponse, get_llm_adapter
from context_cache.llm_config import LLMConfig, LLMConfigStore
from context_cache.tool_router import RouteResult, ToolRouter

logger = logging.getLogger("contextcache.orchestrator")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfidenceLevel(Enum):
    HIGH = "high"
    LOW = "low"
    NO_TOOL = "no_tool"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.2
    max_tool_calls: int = 10
    timeout_s: float = 120.0
    top_k_low_confidence: int = 5
    synthesis_temperature: float = 0.3
    max_synthesis_tokens: int = 1024
    enable_thinking: bool = False  # LLM thinking/reasoning mode (slower but deeper)

    @classmethod
    def from_dict(cls, d: dict) -> OrchestratorConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Tool Executors
# ---------------------------------------------------------------------------


class ToolExecutor(ABC):
    """Abstract base for tool execution."""

    @abstractmethod
    async def execute(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool and return the result."""
        ...


class MockExecutor(ToolExecutor):
    """Returns mock results for testing."""

    async def execute(self, tool_name: str, arguments: dict) -> dict:
        return {
            "tool": tool_name,
            "status": "success",
            "data": {
                "message": f"Mock result from {tool_name}",
                "arguments_received": arguments,
            },
        }


class HttpExecutor(ToolExecutor):
    """Executes tools by POSTing to an HTTP endpoint."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def execute(self, tool_name: str, arguments: dict) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self.base_url,
                json={"tool": tool_name, "arguments": arguments},
            )
            return resp.json()


class CallableExecutor(ToolExecutor):
    """Executes tools by calling registered Python callables."""

    def __init__(self):
        self._handlers: dict[str, Callable] = {}

    def register(self, tool_name: str, handler: Callable):
        self._handlers[tool_name] = handler

    async def execute(self, tool_name: str, arguments: dict) -> dict:
        handler = self._handlers.get(tool_name)
        if handler is None:
            return {"error": f"No handler registered for tool '{tool_name}'"}
        result = handler(**arguments)
        if asyncio.iscoroutine(result):
            result = await result
        return result if isinstance(result, dict) else {"result": result}


def get_executor(spec: str) -> ToolExecutor:
    """Factory for tool executors."""
    if spec == "mock":
        return MockExecutor()
    if spec.startswith("http"):
        return HttpExecutor(spec)
    return MockExecutor()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    """Record of a single tool call in the orchestrator loop."""

    tool_name: str
    arguments: dict
    result: dict
    route_confidence: float
    confidence_level: ConfidenceLevel
    route_ms: float
    execute_ms: float
    llm_ms: float


@dataclass
class OrchestratorResult:
    """Complete result from the orchestrator pipeline."""

    final_response: str
    tool_calls: list[ToolCallRecord]
    total_ms: float
    route_ms: float
    llm_ms: float
    confidence: float
    confidence_level: str
    num_tool_calls: int
    timings: dict
    thinking: str | None = None  # LLM reasoning trace when thinking is enabled


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


PARAM_EXTRACTION_SYSTEM = (
    "Extract parameter values from the user's request. "
    "Respond with ONLY a valid JSON object. No explanation."
)

SYNTHESIS_SYSTEM = (
    "You are a helpful assistant. The user asked a question, a tool was called "
    "to get the data, and you now need to provide a clear, helpful response "
    "based on the tool's result. Be concise and natural."
)

MULTI_TOOL_SYSTEM = (
    "You are a helpful assistant with access to tools. "
    "When you need to call a tool, respond with a JSON block:\n"
    '{"tool_request": {"name": "tool_name", "arguments": {...}}}\n\n'
    "When you have all the data you need, respond normally to the user. "
    "Do NOT include tool_request in your final response."
)


class Orchestrator:
    """Confidence-gated route-first orchestrator.

    Args:
        router: ToolRouter instance for local model tool routing.
        llm_config_store: Server-side LLM credential store.
        config: OrchestratorConfig with thresholds and limits.
    """

    def __init__(
        self,
        router: ToolRouter,
        llm_config_store: LLMConfigStore | None = None,
        config: OrchestratorConfig | None = None,
    ):
        self.router = router
        self.llm_config_store = llm_config_store or LLMConfigStore()
        self.config = config or OrchestratorConfig()

    async def process_query(
        self,
        domain_id: str,
        query: str,
        tool_executor: ToolExecutor | str = "mock",
        llm_api_key: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_extra_headers: dict[str, str] | None = None,
        llm_system_prompt: str | None = None,
        enable_thinking: bool | None = None,
    ) -> OrchestratorResult:
        """Run the full orchestrator pipeline.

        Args:
            enable_thinking: Override config's enable_thinking for this request.
                When True, the LLM uses thinking/reasoning mode for deeper
                analysis. Slower but can improve quality for complex queries.
                The reasoning trace is returned in result.thinking.
        """
        t0 = time.perf_counter()
        timings: dict[str, Any] = {}
        tool_calls: list[ToolCallRecord] = []
        thinking_parts: list[str] = []

        # Resolve thinking mode: per-request override > config default
        think = enable_thinking if enable_thinking is not None else self.config.enable_thinking

        # Resolve executor
        if isinstance(tool_executor, str):
            tool_executor = get_executor(tool_executor)

        # Resolve LLM adapter
        adapter = self._resolve_llm_adapter(
            domain_id,
            llm_api_key,
            llm_provider,
            llm_model,
            llm_base_url,
            llm_extra_headers,
        )

        # Step 1: Route + extract params (local model, cached KV)
        route_result = await self.router.route(
            domain_id, query, top_k=self.config.top_k_low_confidence
        )
        timings["route_ms"] = route_result.route_ms
        timings["route_confidence"] = route_result.confidence
        timings["route_top_candidates"] = route_result.top_candidates[:5]

        # Step 2: Confidence gate
        level = self._classify_confidence(route_result.confidence)
        timings["confidence_level"] = level.value

        # NO_TOOL: Direct conversational response (no tool involved)
        if level == ConfidenceLevel.NO_TOOL:
            messages = [{"role": "user", "content": query}]
            t_llm = time.perf_counter()
            llm_resp = await adapter.complete(
                messages, system_prompt=llm_system_prompt
            )
            llm_ms = (time.perf_counter() - t_llm) * 1000
            timings["llm_ms"] = round(llm_ms, 1)
            timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return OrchestratorResult(
                final_response=llm_resp.content,
                tool_calls=[],
                total_ms=timings["total_ms"],
                route_ms=timings["route_ms"],
                llm_ms=timings["llm_ms"],
                confidence=route_result.confidence,
                confidence_level=level.value,
                num_tool_calls=0,
                timings=timings,
            )

        # Step 3: LLM param extraction — minimal prompt (just param names/types)
        # NOTE: Param extraction ALWAYS uses think=false. It's a simple JSON
        # extraction task where thinking adds 12x latency and hurts accuracy
        # (models over-think and output explanation instead of clean JSON).
        tool_name = route_result.tool_name
        param_prompt = self._build_minimal_param_prompt(
            domain_id, tool_name, query, think=False
        )

        t_llm = time.perf_counter()
        arguments, _ = await self._extract_params(
            adapter, param_prompt, llm_base_url, llm_model, think=False
        )
        param_ms = (time.perf_counter() - t_llm) * 1000
        timings["param_extraction_ms"] = round(param_ms, 1)

        # Step 4: Execute tool
        t_exec = time.perf_counter()
        tool_result = await tool_executor.execute(tool_name, arguments)
        exec_ms = (time.perf_counter() - t_exec) * 1000
        timings["execute_ms"] = round(exec_ms, 1)

        tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result=tool_result,
                route_confidence=route_result.confidence,
                confidence_level=level,
                route_ms=route_result.route_ms,
                execute_ms=round(exec_ms, 1),
                llm_ms=round(param_ms, 1),
            )
        )

        # Step 5: LLM synthesis — just query + result, NO schema
        conversation = self._build_synthesis_messages(
            query, tool_name, arguments, tool_result, llm_system_prompt
        )
        t_llm = time.perf_counter()
        synth_content, synth_thinking = await self._synthesize(
            adapter, conversation, SYNTHESIS_SYSTEM,
            llm_base_url, llm_model,
            max_tokens=self.config.max_synthesis_tokens,
            temperature=self.config.synthesis_temperature,
            think=think,
        )
        synthesis_ms = (time.perf_counter() - t_llm) * 1000
        timings["synthesis_ms"] = round(synthesis_ms, 1)
        if synth_thinking:
            thinking_parts.append(f"[synthesis]\n{synth_thinking}")

        # Step 6: Multi-tool loop — if LLM needs more data, it requests
        # another tool call, we route+execute locally and feed result back
        total_tool_calls = 1
        while total_tool_calls < self.config.max_tool_calls:
            elapsed = time.perf_counter() - t0
            if elapsed > self.config.timeout_s:
                logger.warning("Orchestrator timeout after %.1fs", elapsed)
                break

            parsed_name, parsed_args = self._parse_tool_request(synth_content)
            if not parsed_name:
                break  # LLM returned final text answer

            # LLM requested another tool — route via local model
            sub_route = await self.router.route(domain_id, parsed_name)
            actual_tool = sub_route.tool_name
            actual_args = sub_route.arguments if sub_route.arguments else parsed_args

            t_exec = time.perf_counter()
            sub_result = await tool_executor.execute(actual_tool, actual_args)
            sub_exec_ms = (time.perf_counter() - t_exec) * 1000

            tool_calls.append(
                ToolCallRecord(
                    tool_name=actual_tool,
                    arguments=actual_args,
                    result=sub_result,
                    route_confidence=sub_route.confidence,
                    confidence_level=self._classify_confidence(sub_route.confidence),
                    route_ms=sub_route.route_ms,
                    execute_ms=round(sub_exec_ms, 1),
                    llm_ms=0.0,
                )
            )
            total_tool_calls += 1

            # Feed tool result back to LLM conversation
            conversation.append({"role": "assistant", "content": synth_content})
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool '{actual_tool}' returned:\n"
                        f"```json\n{json.dumps(sub_result, indent=2)}\n```\n\n"
                        "Continue with the next step, or provide your final response."
                    ),
                }
            )

            t_llm = time.perf_counter()
            synth_content, loop_thinking = await self._synthesize(
                adapter, conversation, MULTI_TOOL_SYSTEM,
                llm_base_url, llm_model,
                max_tokens=self.config.max_synthesis_tokens,
                temperature=self.config.synthesis_temperature,
                think=think,
            )
            loop_ms = (time.perf_counter() - t_llm) * 1000
            timings[f"loop_{total_tool_calls}_ms"] = round(loop_ms, 1)
            if loop_thinking:
                thinking_parts.append(f"[loop_{total_tool_calls}]\n{loop_thinking}")

        total_ms = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = round(total_ms, 1)
        timings["thinking_enabled"] = think

        return OrchestratorResult(
            final_response=synth_content,
            tool_calls=tool_calls,
            total_ms=round(total_ms, 1),
            route_ms=route_result.route_ms,
            llm_ms=round(timings.get("synthesis_ms", 0), 1),
            confidence=route_result.confidence,
            confidence_level=level.value,
            num_tool_calls=total_tool_calls,
            timings=timings,
            thinking="\n\n".join(thinking_parts) if thinking_parts else None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_confidence(self, confidence: float) -> ConfidenceLevel:
        """Apply confidence thresholds."""
        if confidence >= self.config.high_confidence_threshold:
            return ConfidenceLevel.HIGH
        if confidence >= self.config.low_confidence_threshold:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.NO_TOOL

    def _resolve_llm_adapter(
        self,
        domain_id: str,
        llm_api_key: str | None,
        llm_provider: str | None,
        llm_model: str | None,
        llm_base_url: str | None,
        llm_extra_headers: dict[str, str] | None,
    ) -> LLMAdapter:
        """Resolve LLM adapter: request params -> config store -> error."""
        # 1. Explicit request params
        if llm_api_key and llm_provider:
            return get_llm_adapter(
                llm_provider,
                api_key=llm_api_key,
                model=llm_model,
                base_url=llm_base_url,
                extra_headers=llm_extra_headers,
            )

        # 2. Per-domain config from store
        if self.llm_config_store:
            config = self.llm_config_store.resolve(domain_id)
            if config and config.is_configured:
                return get_llm_adapter(
                    config.provider,
                    api_key=config.api_key,
                    model=llm_model or config.model,
                    base_url=llm_base_url or config.base_url,
                    extra_headers=llm_extra_headers or config.extra_headers or None,
                )

        # 3. Request-level key without provider (default to openai)
        if llm_api_key:
            return get_llm_adapter(
                llm_provider or "openai",
                api_key=llm_api_key,
                model=llm_model,
                base_url=llm_base_url,
                extra_headers=llm_extra_headers,
            )

        raise ValueError(
            "No LLM credentials configured. Pass llm_api_key/llm_provider in the request, "
            "or configure server-side via LLMConfigStore."
        )

    async def _ollama_chat(
        self,
        messages: list[dict],
        llm_base_url: str,
        llm_model: str | None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        think: bool = False,
    ) -> tuple[str, str | None] | None:
        """Call Ollama's native /api/chat.

        Args:
            think: Enable thinking/reasoning mode. When True, the model
                reasons before responding. Ollama returns separate
                'thinking' and 'content' fields in the response.

        Returns:
            (content, thinking) tuple, or None if the call fails.
            thinking is None when think=False or model doesn't reason.
        """
        import httpx

        ollama_url = llm_base_url.split("/v1")[0].rstrip("/")
        try:
            # When thinking is enabled, allow more tokens for the reasoning
            predict_tokens = max_tokens * 4 if think else max_tokens
            async with httpx.AsyncClient(timeout=120.0 if think else 60.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": llm_model or "qwen3.5:9b",
                        "messages": messages,
                        "stream": False,
                        "think": think,
                        "options": {
                            "temperature": temperature,
                            "num_predict": predict_tokens,
                        },
                    },
                )
                data = resp.json()
                msg = data.get("message", {})
                content = msg.get("content", "")
                thinking = msg.get("thinking") if think else None
                return content, thinking
        except Exception as e:
            logger.warning("Ollama native API failed: %s, falling back", e)
            return None

    def _is_ollama(self, llm_base_url: str | None) -> bool:
        """Detect if the LLM endpoint is Ollama."""
        base_url = llm_base_url or ""
        return "11434" in base_url or "ollama" in base_url.lower()

    async def _extract_params(
        self,
        adapter: LLMAdapter,
        param_prompt: str,
        llm_base_url: str | None,
        llm_model: str | None,
        think: bool = False,
    ) -> tuple[dict, str | None]:
        """Extract parameters from the user's request.

        Returns:
            (arguments_dict, thinking_trace_or_none)
        """
        if self._is_ollama(llm_base_url):
            result = await self._ollama_chat(
                messages=[
                    {"role": "system", "content": PARAM_EXTRACTION_SYSTEM},
                    {"role": "user", "content": param_prompt},
                ],
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                max_tokens=256,
                temperature=0.0,
                think=think,
            )
            if result is not None:
                content, thinking = result
                return self._parse_json_response(content), thinking

        # Fallback: standard LLM adapter
        llm_resp = await adapter.complete(
            [{"role": "user", "content": param_prompt}],
            system_prompt=PARAM_EXTRACTION_SYSTEM,
            max_tokens=256,
            temperature=0.0,
        )
        content = llm_resp.content
        # For non-Ollama providers with thinking, extract content after </think>
        thinking = None
        if think and "<think>" in content:
            thinking, content = self._split_thinking(content)
        return self._parse_json_response(content), thinking

    async def _synthesize(
        self,
        adapter: LLMAdapter,
        messages: list[dict],
        system_prompt: str,
        llm_base_url: str | None,
        llm_model: str | None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        think: bool = False,
    ) -> tuple[str, str | None]:
        """Synthesize a response via the LLM.

        Returns:
            (content, thinking_trace_or_none)
        """
        if self._is_ollama(llm_base_url):
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            result = await self._ollama_chat(
                messages=full_messages,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                max_tokens=max_tokens,
                temperature=temperature,
                think=think,
            )
            if result is not None:
                return result  # (content, thinking)

        # Fallback: standard LLM adapter
        llm_resp = await adapter.complete(
            messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = llm_resp.content
        thinking = None
        if think and "<think>" in content:
            thinking, content = self._split_thinking(content)
        return content, thinking

    def _build_minimal_param_prompt(
        self, domain_id: str, tool_name: str, query: str, think: bool = False
    ) -> str:
        """Build a minimal param extraction prompt — just param names/types.

        Instead of sending the full tool schema, we send only:
          Tool: check_inventory
          Params: product (string, required), warehouse (string, optional)
          Request: "Do we have iPhone 16 Pro in stock?"

        This is ~3 lines vs a full JSON schema. The LLM just outputs the values.
        When think=False, prepends /no_think to suppress reasoning.
        """
        domain = self.router._domains.get(domain_id)
        if not domain:
            return f'Extract parameters from: "{query}"'

        # Find the matched tool's schema
        schema = None
        for t in domain.tools:
            func = t.get("function", t)
            if func.get("name") == tool_name:
                schema = func
                break

        if not schema or "parameters" not in schema:
            return f'Extract parameters from: "{query}"'

        # Build compact param description
        props = schema["parameters"].get("properties", {})
        required = set(schema["parameters"].get("required", []))
        param_parts = []
        for name, info in props.items():
            ptype = info.get("type", "string")
            req = "required" if name in required else "optional"
            desc = info.get("description", "")
            param_parts.append(f"  {name} ({ptype}, {req}): {desc}")

        params_text = "\n".join(param_parts)
        # Only suppress thinking when think=False
        prefix = "" if think else "/no_think\n"
        return (
            f"{prefix}"
            f"Tool: {tool_name}\n"
            f"Parameters:\n{params_text}\n\n"
            f'User request: "{query}"\n\n'
            "Return a JSON object with the parameter values."
        )

    def _split_thinking(self, text: str) -> tuple[str, str]:
        """Split <think>...</think> from content for non-Ollama providers.

        Returns (thinking_text, content_text).
        """
        match = re.search(r"<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", text

    def _parse_json_response(self, text: str) -> dict:
        """Parse a JSON object from LLM response text."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {}

    def _build_synthesis_messages(
        self,
        query: str,
        tool_name: str,
        arguments: dict,
        tool_result: dict,
        user_system_prompt: str | None,
    ) -> list[dict]:
        """Build messages for LLM to synthesize a final answer from tool result."""
        return [
            {
                "role": "user",
                "content": (
                    f"The user asked: {query}\n\n"
                    f"Tool '{tool_name}' was called with: {json.dumps(arguments)}\n\n"
                    f"Tool result:\n```json\n{json.dumps(tool_result, indent=2)}\n```\n\n"
                    "Provide a clear, helpful response to the user based on this data."
                ),
            }
        ]

    def _parse_tool_request(self, text: str) -> tuple[str | None, dict]:
        """Parse a tool_request from LLM response in multi-tool loop.

        Looks for: {"tool_request": {"name": "...", "arguments": {...}}}
        """
        try:
            match = re.search(
                r'\{\s*"tool_request"\s*:\s*\{[^}]*\}\s*\}', text, re.DOTALL
            )
            if match:
                parsed = json.loads(match.group())
                req = parsed.get("tool_request", {})
                return req.get("name"), req.get("arguments", {})
        except (json.JSONDecodeError, AttributeError):
            pass
        return None, {}

