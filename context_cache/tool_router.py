"""Async-safe tool router using LlamaCppEngine with KV state caching.

Wraps the in-process llama.cpp engine for fast tool name routing (~550ms).
Each domain gets a cached KV state (system prompt + tool schemas prefilled),
enabling warm queries via state restore instead of full re-prefill.

Thread safety: All engine operations are guarded by an asyncio.Lock and
dispatched to a thread pool executor since LlamaCppEngine is not thread-safe.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from context_cache.llama_cpp_engine import LlamaCppEngine

logger = logging.getLogger("contextcache.router")

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise tool-routing assistant. Your ONLY job is to select the "
    "single best-matching tool for the user's request. You MUST respond with "
    "exactly one tool call - no text, no explanation, no alternatives.\n\n"
    "# How to Select the Right Tool\n\n"
    "1. Identify the PRIMARY ACTION VERB in the user's request.\n"
    "2. Match that action verb to the tool name, ignoring nouns that look like other tools.\n"
    "3. The verb determines the tool. Nouns in the query are just context."
)


@dataclass
class DomainState:
    """Cached state for one registered domain."""

    domain_id: str
    tools: list[dict]
    tool_names: list[str]
    system_prompt: str
    state_data: bytes
    prefix_token_count: int
    state_size_mb: float
    prefill_ms: float
    registered_at: float
    query_count: int = 0


@dataclass
class RouteResult:
    """Result from tool routing."""

    tool_name: str
    arguments: dict
    confidence: float
    top_candidates: list[tuple[str, float]]
    route_ms: float
    restore_ms: float
    prefill_ms: float
    gen_ms: float
    raw_tool_call: str = ""


class ToolRouter:
    """Async-safe tool router backed by LlamaCppEngine.

    Args:
        model_path: Path to the GGUF model file.
        n_ctx: Context window size (tokens).
        n_threads: CPU threads for inference.
        n_batch: Batch size for prefill.
        system_prompt: Default system prompt for tool routing.
        max_domains: Maximum number of cached domains (LRU eviction).
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 16384,
        n_threads: int = 4,
        n_batch: int = 2048,
        system_prompt: str | None = None,
        max_domains: int = 20,
    ):
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_batch = n_batch
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._max_domains = max_domains

        self._engine: Optional[LlamaCppEngine] = None
        self._domains: dict[str, DomainState] = {}
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="router")
        self._loaded = False

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def load_model(self) -> float:
        """Load the GGUF model. Returns load time in ms."""
        loop = asyncio.get_event_loop()
        load_ms = await loop.run_in_executor(self._executor, self._sync_load_model)
        self._loaded = True
        return load_ms

    async def register_tools(
        self,
        domain_id: str,
        tools: list[dict],
        system_prompt: str | None = None,
    ) -> dict:
        """Register a tool catalog for a domain.

        Builds the chat template prefix, prefills it, and saves the KV state.
        This is the expensive operation (~44-106s for 50 tools on CPU).

        Returns:
            Dict with {domain_id, num_tools, prefix_tokens, state_size_mb, prefill_ms}.
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._sync_register,
                domain_id,
                tools,
                system_prompt,
            )

    async def route(
        self,
        domain_id: str,
        query: str,
        top_k: int = 5,
    ) -> RouteResult:
        """Route a query to the best-matching tool.

        Restores cached KV state, prefills the query suffix, samples the
        tool name greedily, and computes confidence from logits.
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor, self._sync_route, domain_id, query, top_k
            )

    async def unregister_domain(self, domain_id: str) -> bool:
        """Remove a domain's cached state."""
        async with self._lock:
            if domain_id in self._domains:
                del self._domains[domain_id]
                return True
            return False

    async def list_domains(self) -> list[dict]:
        """List all registered domains with summary info."""
        return [
            {
                "domain_id": ds.domain_id,
                "num_tools": len(ds.tools),
                "tool_names": ds.tool_names[:10],
                "state_size_mb": round(ds.state_size_mb, 1),
                "prefix_tokens": ds.prefix_token_count,
                "query_count": ds.query_count,
                "registered_at": ds.registered_at,
            }
            for ds in self._domains.values()
        ]

    async def close(self):
        """Shutdown engine and executor."""
        async with self._lock:
            if self._engine:
                self._engine.close()
                self._engine = None
            self._loaded = False
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Prompt building (Qwen3.5 chat template)
    # ------------------------------------------------------------------

    def _build_prefix_prompt(self, system_prompt: str, tools: list[dict]) -> str:
        """Build the Qwen3.5 chat template prefix (system + tools).

        This is the static part that gets prefilled once and cached.
        """
        tool_strs = "\n".join(json.dumps(t) for t in tools)
        return (
            f"<|im_start|>system\n"
            f"{system_prompt}\n\n"
            f"# Tools\n\n"
            f"You have access to the following functions:\n\n"
            f"<tools>\n"
            f"{tool_strs}\n"
            f"</tools><|im_end|>\n"
        )

    def _build_query_suffix(self, query: str) -> str:
        """Build the per-query suffix with forced tool_call prefix.

        Includes /no_think to disable Qwen3.5 thinking mode.
        """
        return (
            f"<|im_start|>user\n"
            f"/no_think\n"
            f"{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<tool_call>\n"
            f'{{"name": "'
        )

    # ------------------------------------------------------------------
    # Synchronous engine operations (run in executor thread)
    # ------------------------------------------------------------------

    def _sync_load_model(self) -> float:
        """Load model in the executor thread."""
        t0 = time.perf_counter()
        self._engine = LlamaCppEngine(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            n_batch=self._n_batch,
            verbose=False,
        )
        load_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Model loaded: %s (%.0fms, vocab=%d)",
            self._model_path,
            load_ms,
            self._engine.n_vocab,
        )
        return load_ms

    def _sync_register(
        self,
        domain_id: str,
        tools: list[dict],
        system_prompt: str | None,
    ) -> dict:
        """Synchronous registration: prefill + save state."""
        engine = self._engine
        if engine is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompt = system_prompt or self._system_prompt
        prefix_text = self._build_prefix_prompt(prompt, tools)
        prefix_tokens = engine.tokenize(prefix_text, add_bos=True)

        logger.info(
            "Registering domain '%s': %d tools, %d prefix tokens",
            domain_id,
            len(tools),
            len(prefix_tokens),
        )

        # Cold prefill
        t0 = time.perf_counter()
        engine.prefill(prefix_tokens, start_pos=0)
        prefill_ms = (time.perf_counter() - t0) * 1000

        # Save state
        t0 = time.perf_counter()
        state_data = engine.save_state()
        save_ms = (time.perf_counter() - t0) * 1000
        state_mb = len(state_data) / (1024 * 1024)

        logger.info(
            "Domain '%s' registered: prefill=%.0fms, save=%.0fms, state=%.1fMB",
            domain_id,
            prefill_ms,
            save_ms,
            state_mb,
        )

        # Extract tool names
        tool_names = []
        for t in tools:
            func = t.get("function", t)
            tool_names.append(func.get("name", "unknown"))

        # Store domain state
        self._domains[domain_id] = DomainState(
            domain_id=domain_id,
            tools=tools,
            tool_names=tool_names,
            system_prompt=prompt,
            state_data=state_data,
            prefix_token_count=len(prefix_tokens),
            state_size_mb=state_mb,
            prefill_ms=prefill_ms,
            registered_at=time.time(),
        )

        # LRU eviction if over limit
        if len(self._domains) > self._max_domains:
            oldest_id = min(
                self._domains,
                key=lambda k: self._domains[k].registered_at,
            )
            del self._domains[oldest_id]
            logger.info("Evicted domain '%s' (over max_domains=%d)", oldest_id, self._max_domains)

        return {
            "domain_id": domain_id,
            "num_tools": len(tools),
            "prefix_tokens": len(prefix_tokens),
            "state_size_mb": round(state_mb, 1),
            "prefill_ms": round(prefill_ms, 1),
        }

    def _sync_route(self, domain_id: str, query: str, top_k: int) -> RouteResult:
        """Synchronous routing: restore state, prefill query, sample tool name.

        The 2B model is optimized for tool selection (100% accuracy, ~500ms).
        Parameter extraction is handled separately by the orchestrator.
        """
        engine = self._engine
        if engine is None:
            raise RuntimeError("Model not loaded.")

        domain = self._domains.get(domain_id)
        if domain is None:
            raise ValueError(f"Domain '{domain_id}' not registered.")

        domain.query_count += 1

        # 1. Restore KV state
        t0 = time.perf_counter()
        engine.load_state(domain.state_data)
        restore_ms = (time.perf_counter() - t0) * 1000

        # 2. Tokenize query suffix
        query_suffix = self._build_query_suffix(query)
        query_tokens = engine.tokenize(query_suffix, add_bos=False)

        # 3. Prefill query tokens at offset
        t0 = time.perf_counter()
        engine.prefill(query_tokens, start_pos=domain.prefix_token_count)
        prefill_ms = (time.perf_counter() - t0) * 1000

        # 4. Compute confidence from logits (before generation)
        confidence, top_token_probs = self._compute_confidence(top_k)

        # 5. Generate tool name (stop at closing quote)
        t0 = time.perf_counter()
        gen_tokens = []
        pos = domain.prefix_token_count + len(query_tokens)

        for _ in range(64):  # max tokens for a tool name
            tok = engine.sample_greedy(pos)
            if tok == engine.eos_token:
                break
            piece = engine.detokenize([tok])
            if '"' in piece:
                gen_tokens.append(tok)
                break
            gen_tokens.append(tok)

            # Feed token back
            batch = engine._make_batch([tok], pos, logits_last=True)
            rc = engine.lib.llama_decode(engine.ctx, batch)
            engine.lib.llama_batch_free(batch)
            if rc != 0:
                logger.warning("decode failed during generation: rc=%d", rc)
                break
            pos += 1

        gen_ms = (time.perf_counter() - t0) * 1000

        # 6. Extract tool name
        gen_text = engine.detokenize(gen_tokens)
        tool_name = gen_text.strip().split('"')[0].strip()

        # 7. Map top tokens to tool name candidates
        top_candidates = self._match_token_to_tools(top_token_probs, domain.tool_names)

        total_ms = restore_ms + prefill_ms + gen_ms

        return RouteResult(
            tool_name=tool_name,
            arguments={},
            confidence=confidence,
            top_candidates=top_candidates,
            route_ms=round(total_ms, 1),
            restore_ms=round(restore_ms, 1),
            prefill_ms=round(prefill_ms, 1),
            gen_ms=round(gen_ms, 1),
        )

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(self, top_k: int = 5) -> tuple[float, list[tuple[int, float]]]:
        """Compute confidence from logits at the last decoded position.

        Returns (max_probability, [(token_id, probability), ...]).
        """
        engine = self._engine
        logits_ptr = engine.lib.llama_get_logits_ith(engine.ctx, -1)

        # Copy to numpy (avoid dangling pointer after next decode)
        logits_np = np.ctypeslib.as_array(
            ctypes.cast(logits_ptr, ctypes.POINTER(ctypes.c_float)),
            shape=(engine.n_vocab,),
        ).copy()

        # Softmax with numerical stability
        shifted = logits_np - logits_np.max()
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum()

        # Top-K
        top_indices = np.argpartition(probs, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]

        max_prob = float(probs[top_indices[0]])
        top_token_probs = [(int(idx), float(probs[idx])) for idx in top_indices]

        return max_prob, top_token_probs

    def _match_token_to_tools(
        self,
        top_tokens: list[tuple[int, float]],
        tool_names: list[str],
    ) -> list[tuple[str, float]]:
        """Map top token IDs to matching tool names with probabilities.

        For each top token, detokenize it and check which tool names
        start with that text. Aggregate probabilities for tools sharing
        a common prefix token.
        """
        engine = self._engine
        tool_probs: dict[str, float] = {}

        for tok_id, prob in top_tokens:
            piece = engine.detokenize([tok_id]).strip()
            if not piece:
                continue
            for name in tool_names:
                if name.startswith(piece) or name.lower().startswith(piece.lower()):
                    tool_probs[name] = tool_probs.get(name, 0.0) + prob

        # Sort by probability descending
        sorted_tools = sorted(tool_probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_tools[:10]
