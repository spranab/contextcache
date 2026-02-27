"""ContextCacheModel: Qwen3-8B with NoPE KV capture for position-independent caching.

Three-phase architecture:
1. COMPILE: Forward a context block → capture pre-RoPE keys → store with content hash
2. LINK: Load cached NoPE KV → apply deferred RoPE at correct positions → compose
3. EXECUTE: Forward user query with composed KV cache → generate response

The monkey-patch intercepts Qwen3Attention.forward() after k_norm (pre-RoPE keys)
but before apply_rotary_pos_emb, storing position-independent key states.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

from context_cache.cache_config import ContextCacheConfig
from context_cache.kv_store import CachedContextKV, ContextKVStore
from context_cache.model_adapter import ModelAdapter, get_adapter
from context_cache.rope_utils import apply_rope, build_rope_cache


class ContextCacheModel:
    """Qwen3-8B with NoPE KV capture for position-independent context caching."""

    def __init__(self, config: ContextCacheConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        print("Loading model...")
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create model adapter (handles prompt formatting, layer access, stop tokens)
        if config.model.adapter == "auto":
            self.adapter: ModelAdapter = get_adapter(config.model.model_name, self.model, self.tokenizer)
        else:
            from context_cache.model_adapter import ADAPTER_REGISTRY
            adapter_cls = ADAPTER_REGISTRY.get(config.model.adapter)
            if adapter_cls is None:
                raise ValueError(
                    f"Unknown adapter '{config.model.adapter}'. "
                    f"Available: {list(ADAPTER_REGISTRY.keys())}"
                )
            self.adapter = adapter_cls(self.model, self.tokenizer)

        # Extract model architecture params via adapter
        self.num_layers = self.adapter.num_layers
        self.num_kv_heads = self.adapter.num_kv_heads
        self.head_dim = self.adapter.head_dim

        # Build RoPE cos/sin tables
        rope_theta = config.rope.theta if config.rope.theta != 1_000_000.0 else self.adapter.rope_theta
        self.rope_cos, self.rope_sin = build_rope_cache(
            max_seq_len=config.rope.max_position,
            head_dim=self.head_dim,
            rope_theta=rope_theta,
            device=self.device,
            dtype=torch.bfloat16 if config.model.bnb_4bit_compute_dtype == "bfloat16" else torch.float16,
        )

        # Initialize KV store
        self.kv_store = ContextKVStore(
            cache_dir=config.cache.cache_dir,
            num_layers=self.num_layers,
        )

        # Monkey-patch attention layers for NoPE capture (optional, adapter-dependent)
        self._nope_key_buffer: dict[int, Tensor] = {}
        self._capture_nope = False
        self._nope_capture_counter = 0
        self._nope_capture_state = {
            "enabled": False,
            "counter": 0,
            "buffer": self._nope_key_buffer,
        }
        try:
            self.adapter.monkey_patch_rope_capture(self._nope_capture_state)
            self._nope_patched = True
        except NotImplementedError:
            self._nope_patched = False

        # Map layer index → device (handles multi-GPU device_map="auto")
        self._layer_devices = self.adapter.get_layer_devices()

        # Group cache for tool-set-level caching (generate_group_cached)
        self._group_cache: dict[str, tuple[DynamicCache, int]] = {}

        # Stop tokens for generation (via adapter)
        self._stop_token_ids = self.adapter.get_stop_token_ids()

        print(
            f"ContextCacheModel ready: {self.num_layers} layers, "
            f"{self.num_kv_heads} KV heads, head_dim={self.head_dim}, "
            f"adapter={self.adapter.__class__.__name__}"
        )

    def _load_model(self) -> AutoModelForCausalLM:
        """Load Qwen3-8B with 4-bit quantization."""
        cfg = self.config.model
        quant_config = None
        if cfg.load_in_4bit:
            compute_dtype = (
                torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
            )
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            )

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # Need eager for monkey-patch compatibility
        )
        model.eval()
        return model

    # ========================
    # COMPILE PHASE
    # ========================

    @torch.no_grad()
    def compile_context(
        self,
        text: str,
        name: str,
        content_type: str = "tool_schema",
        force: bool = False,
        prefix_text: str | None = None,
    ) -> CachedContextKV:
        """Compile a context block into NoPE KV states.

        When prefix_text is provided, the content is compiled with the prefix
        as context (e.g., system prompt + tool instructions), producing NoPE
        keys that capture cross-attention with the prefix. Only the content
        portion's KV is stored — the prefix is discarded (it gets computed
        fresh at link time).

        When prefix_text is None, falls back to dummy BOS tokens (LegoLink-0).

        Args:
            text: The content text (tool schema JSON, code, document, etc.)
            name: Human-readable identifier (tool name, file path, etc.)
            content_type: Category for indexing ("tool_schema", "code_file", etc.)
            force: If True, recompile even if cached.
            prefix_text: Optional prefix text to use as context during compilation.

        Returns:
            CachedContextKV with NoPE keys and values.
        """
        # Check cache first
        if not force and self.kv_store.has(text):
            cached = self.kv_store.get(text, device=str(self.device))
            if cached is not None:
                return cached

        content_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if prefix_text is not None:
            # Prefix-aware compilation: [prefix_tokens] + [content_tokens]
            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            input_ids = torch.tensor([prefix_ids + content_ids], device=self.device)
            prefix_len = len(prefix_ids)
        else:
            # Fallback: dummy BOS tokens as context
            num_dummy = self.config.cache.num_dummy_bos
            bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            dummy_ids = [bos_id] * num_dummy
            input_ids = torch.tensor([dummy_ids + content_ids], device=self.device)
            prefix_len = num_dummy

        # Enable NoPE capture
        if not self._nope_patched:
            raise RuntimeError(
                f"NoPE capture not available for {self.adapter.__class__.__name__}. "
                "Use generate_group_cached() instead of compile_context()."
            )
        self._capture_nope = True
        self._nope_capture_state["enabled"] = True
        self._nope_key_buffer.clear()
        self._nope_capture_counter = 0
        self._nope_capture_state["counter"] = 0

        # Forward pass (prefill only, no generation)
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

        # Disable NoPE capture
        self._capture_nope = False
        self._nope_capture_state["enabled"] = False

        # Collect pre-RoPE keys and values (discard prefix positions)
        past_kv = outputs.past_key_values
        nope_keys = []
        values = []

        for layer_idx in range(self.num_layers):
            # Pre-RoPE keys from our buffer — content portion only
            nope_k = self._nope_key_buffer[layer_idx][:, :, prefix_len:, :]
            nope_keys.append(nope_k.cpu())

            # Values from the model's KV cache (values are never rotated)
            v = past_kv.layers[layer_idx].values[:, :, prefix_len:, :]
            values.append(v.cpu())

        num_real_tokens = len(content_ids)

        # Store to persistent cache
        self.kv_store.put(
            name=name,
            text=text,
            content_type=content_type,
            keys=nope_keys,
            values=values,
            num_tokens=num_real_tokens,
        )

        # Clean up
        self._nope_key_buffer.clear()

        cached = CachedContextKV(
            name=name,
            content_hash=self.kv_store.hash_content(text),
            content_type=content_type,
            num_tokens=num_real_tokens,
            keys=nope_keys,
            values=values,
        )
        return cached

    def compile_tool_catalog(self, catalog_path: str | Path) -> list[CachedContextKV]:
        """Compile all tools from a catalog JSON file."""
        with open(catalog_path) as f:
            catalog = json.load(f)

        results = []
        for tool in catalog:
            schema = tool.get("schema", tool)
            name = schema.get("function", {}).get("name", tool.get("tool_id", "unknown"))
            schema_text = json.dumps(schema, separators=(",", ":"))
            cached = self.compile_context(schema_text, name, content_type="tool_schema")
            results.append(cached)
            print(f"  Compiled: {name} ({cached.num_tokens} tokens)")

        return results

    # ========================
    # CHAT TEMPLATE HELPERS (delegated to adapter)
    # ========================

    def _build_tool_chat_parts(
        self,
        tool_schemas: list[str],
        user_query: str,
        system_prompt: str,
    ) -> tuple[str, str]:
        """Build prompt parts via the model adapter.

        Returns:
            (prefix_text, suffix_text) — prefix is cacheable, suffix is per-request.
        """
        return self.adapter.build_prompt_parts(tool_schemas, user_query, system_prompt)

    def _build_full_prompt(
        self,
        tool_schemas: list[str],
        user_query: str,
        system_prompt: str,
    ) -> str:
        """Build the complete prompt via the model adapter."""
        return self.adapter.build_full_prompt(tool_schemas, user_query, system_prompt)

    # ========================
    # LINK PHASE
    # ========================

    @torch.no_grad()
    def link_contexts(
        self,
        context_texts: list[str],
        system_prompt: str | None = None,
    ) -> tuple[DynamicCache, int]:
        """Compose cached NoPE KV for selected contexts with deferred RoPE.

        Args:
            context_texts: List of content texts to compose (must be pre-compiled).
            system_prompt: Optional system prompt (computed fresh, not cached).

        Returns:
            (kv_cache, total_prefix_len) where:
            - kv_cache: DynamicCache with system prompt + all tool KV
            - total_prefix_len: total number of prefix tokens
        """
        kv_cache = DynamicCache()
        current_pos = 0

        # 1. Compute system prompt / prefix KV fresh (it gets positions [0, S))
        if system_prompt:
            sys_ids = self.tokenizer.encode(system_prompt, add_special_tokens=False)
            sys_input = torch.tensor([sys_ids], device=self.device)
            sys_len = len(sys_ids)

            # Forward system prompt through model
            self._capture_nope = False
            sys_outputs = self.model(
                input_ids=sys_input,
                use_cache=True,
                return_dict=True,
            )

            # Extract KV from model's cache (already on correct per-layer devices)
            sys_past = sys_outputs.past_key_values
            for layer_idx in range(self.num_layers):
                k = sys_past.layers[layer_idx].keys
                v = sys_past.layers[layer_idx].values
                kv_cache.update(k.contiguous(), v.contiguous(), layer_idx)

            current_pos = sys_len

        # 2. For each context block, load cached NoPE KV and apply deferred RoPE
        for text in context_texts:
            cached = self.kv_store.get(text, device=str(self.device))
            if cached is None:
                raise ValueError(
                    f"Context not compiled. Call compile_context() first. "
                    f"Hash: {self.kv_store.hash_content(text)[:16]}..."
                )

            # Assign absolute positions for this block
            block_len = cached.num_tokens
            positions = torch.arange(
                current_pos,
                current_pos + block_len,
                device=self.device,
            )

            # Apply deferred RoPE to cached NoPE keys
            for layer_idx in range(self.num_layers):
                layer_device = self._layer_devices[layer_idx]
                nope_k = cached.keys[layer_idx].to(layer_device)
                v = cached.values[layer_idx].to(layer_device)

                # Deferred RoPE: rotate keys to correct absolute positions
                pos_on_dev = positions.to(layer_device)
                cos_on_dev = self.rope_cos.to(layer_device)
                sin_on_dev = self.rope_sin.to(layer_device)
                rotated_k = apply_rope(nope_k, pos_on_dev, cos_on_dev, sin_on_dev)

                # Append to KV cache
                kv_cache.update(rotated_k, v, layer_idx)

            current_pos += block_len

        return kv_cache, current_pos

    # ========================
    # EXECUTE PHASE
    # ========================

    @torch.no_grad()
    def generate(
        self,
        context_texts: list[str],
        user_query: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
    ) -> tuple[str, dict]:
        """Full pipeline: link cached context KV + generate response.

        Args:
            context_texts: Pre-compiled context block texts.
            user_query: The user's question/request.
            system_prompt: Optional system prompt (uses config default if None).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            (generated_text, timing_dict)
        """
        if system_prompt is None:
            system_prompt = self.config.system_prompt

        timings = {}

        # Build chat template parts: prefix (system + tool instructions header)
        # and suffix (tool instructions footer + user query + generation prompt)
        prefix_text, suffix_text = self._build_tool_chat_parts(
            context_texts, user_query, system_prompt,
        )

        # In the full prompt, each tool schema is followed by \n.
        # The cached KV must be compiled with this trailing \n included.
        link_texts = [t + "\n" for t in context_texts]

        # Link phase: prefix KV (fresh) + cached tool KV (with deferred RoPE)
        t0 = time.perf_counter()
        kv_cache, prefix_len = self.link_contexts(link_texts, prefix_text)
        timings["link_ms"] = (time.perf_counter() - t0) * 1000

        # Tokenize the suffix (after-tools instructions + user query + gen prompt)
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)
        suffix_input = torch.tensor([suffix_ids], device=self.device)
        suffix_len = len(suffix_ids)

        # Forward suffix with the composed KV cache
        t0 = time.perf_counter()
        position_ids = torch.arange(
            prefix_len,
            prefix_len + suffix_len,
            device=self.device,
        ).unsqueeze(0)

        cache_position = torch.arange(
            prefix_len,
            prefix_len + suffix_len,
            device=self.device,
        )

        outputs = self.model(
            input_ids=suffix_input,
            position_ids=position_ids,
            past_key_values=kv_cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        timings["prefill_query_ms"] = (time.perf_counter() - t0) * 1000
        prefix_len = prefix_len + suffix_len  # Update total for decode

        # Autoregressive generation
        t0 = time.perf_counter()
        generated_ids = []
        next_token_logits = outputs.logits[:, -1, :]
        kv_cache = outputs.past_key_values
        current_pos = prefix_len

        for _ in range(max_new_tokens):
            next_token = torch.argmax(next_token_logits, dim=-1)
            token_id = next_token.item()

            if token_id in self._stop_token_ids:
                break

            generated_ids.append(token_id)

            pos_ids = torch.tensor([[current_pos]], device=self.device)
            cache_pos = torch.tensor([current_pos], device=self.device)

            outputs = self.model(
                input_ids=next_token.unsqueeze(0),
                position_ids=pos_ids,
                past_key_values=kv_cache,
                cache_position=cache_pos,
                use_cache=True,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            kv_cache = outputs.past_key_values
            current_pos += 1

        timings["decode_ms"] = (time.perf_counter() - t0) * 1000
        timings["num_generated_tokens"] = len(generated_ids)

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return text, timings

    # ========================
    # GROUP CACHED
    # ========================

    @torch.no_grad()
    def generate_group_cached(
        self,
        context_texts: list[str],
        user_query: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
    ) -> tuple[str, dict]:
        """Group-cached generation: cache prefix+tools KV together, forward only suffix.

        On first call with a given tool set: full prefill of prefix+tools, cache the KV.
        On subsequent calls with the same tool set: load cached KV, only forward suffix.

        The cache key is the hash of the sorted tool texts, ensuring that
        the same tool set (regardless of order) always hits the same cache entry.

        This provides near-zero-cost generation for repeated tool sets while
        maintaining full-prefill quality (cross-tool attention is preserved).
        """
        if system_prompt is None:
            system_prompt = self.config.system_prompt

        timings = {}

        prefix_text, suffix_text = self._build_tool_chat_parts(
            context_texts, user_query, system_prompt,
        )

        # Build cache key from sorted tool texts
        group_key = hashlib.sha256(
            "\n".join(sorted(context_texts)).encode()
        ).hexdigest()

        # Check if we have cached KV for this tool group
        # The group cache stores raw tensor lists: ([(k, v), ...], prefix_len)
        group_entry = self._group_cache.get(group_key)

        if group_entry is not None:
            # Cache hit: reconstruct DynamicCache from stored tensors
            t0 = time.perf_counter()
            cached_layers, prefix_len = group_entry
            kv_cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(cached_layers):
                layer_device = self._layer_devices[layer_idx]
                kv_cache.update(k.to(layer_device), v.to(layer_device), layer_idx)
            timings["link_ms"] = (time.perf_counter() - t0) * 1000
            timings["cache_hit"] = True
        else:
            # Cache miss: forward full prefix (system+tools), cache result
            t0 = time.perf_counter()

            # Forward the entire prefix — adapter already split it correctly
            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_input = torch.tensor([prefix_ids], device=self.device)
            prefix_out = self.model(
                input_ids=prefix_input, use_cache=True, return_dict=True,
            )
            kv_cache = prefix_out.past_key_values
            prefix_len = len(prefix_ids)
            timings["link_ms"] = (time.perf_counter() - t0) * 1000
            timings["cache_hit"] = False

            # Cache the KV as raw tensors for future reuse
            cached_layers = []
            for layer_idx in range(self.num_layers):
                k = kv_cache.layers[layer_idx].keys.detach().clone()
                v = kv_cache.layers[layer_idx].values.detach().clone()
                cached_layers.append((k, v))
            self._group_cache[group_key] = (cached_layers, prefix_len)

            # Rebuild a DynamicCache from the model's cache for suffix forward
            new_kv = DynamicCache()
            for layer_idx in range(self.num_layers):
                k = kv_cache.layers[layer_idx].keys
                v = kv_cache.layers[layer_idx].values
                new_kv.update(k.contiguous(), v.contiguous(), layer_idx)
            kv_cache = new_kv

        # Forward suffix
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)
        suffix_input = torch.tensor([suffix_ids], device=self.device)
        suffix_len = len(suffix_ids)

        t0 = time.perf_counter()
        position_ids = torch.arange(
            prefix_len, prefix_len + suffix_len, device=self.device,
        ).unsqueeze(0)
        cache_position = torch.arange(
            prefix_len, prefix_len + suffix_len, device=self.device,
        )

        outputs = self.model(
            input_ids=suffix_input,
            position_ids=position_ids,
            past_key_values=kv_cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        timings["prefill_query_ms"] = (time.perf_counter() - t0) * 1000
        prefix_len = prefix_len + suffix_len

        # Autoregressive generation
        t0 = time.perf_counter()
        generated_ids = []
        next_token_logits = outputs.logits[:, -1, :]
        kv_cache = outputs.past_key_values
        current_pos = prefix_len

        for _ in range(max_new_tokens):
            next_token = torch.argmax(next_token_logits, dim=-1)
            token_id = next_token.item()

            if token_id in self._stop_token_ids:
                break

            generated_ids.append(token_id)

            pos_ids = torch.tensor([[current_pos]], device=self.device)
            cache_pos = torch.tensor([current_pos], device=self.device)

            outputs = self.model(
                input_ids=next_token.unsqueeze(0),
                position_ids=pos_ids,
                past_key_values=kv_cache,
                cache_position=cache_pos,
                use_cache=True,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            kv_cache = outputs.past_key_values
            current_pos += 1

        timings["decode_ms"] = (time.perf_counter() - t0) * 1000
        timings["num_generated_tokens"] = len(generated_ids)

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return text, timings

    # ========================
    # GROUP CACHE PERSISTENCE
    # ========================

    def _group_cache_dir(self) -> Path:
        """Return the directory for persisted group KV caches."""
        d = Path(self.config.cache.cache_dir) / "group_kv"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_group_cache(self, group_key: str, tool_names: list[str] | None = None) -> Path:
        """Save a group KV cache entry to disk for persistence across restarts.

        Args:
            group_key: The SHA256 hash key for this tool group.
            tool_names: Optional list of tool names for metadata.

        Returns:
            Path to the saved cache file.
        """
        entry = self._group_cache.get(group_key)
        if entry is None:
            raise ValueError(f"No in-memory group cache for key {group_key[:16]}...")

        cached_layers, prefix_len = entry
        cache_dir = self._group_cache_dir()

        # Save tensors
        save_data = {
            "prefix_len": prefix_len,
            "keys": [k.cpu() for k, v in cached_layers],
            "values": [v.cpu() for k, v in cached_layers],
        }
        cache_path = cache_dir / f"{group_key}.pt"
        torch.save(save_data, cache_path)

        # Update index
        index_path = cache_dir / "index.json"
        index = {}
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)

        size_mb = cache_path.stat().st_size / (1024 * 1024)
        index[group_key] = {
            "tool_names": tool_names or [],
            "num_tools": len(tool_names) if tool_names else 0,
            "prefix_tokens": prefix_len,
            "cache_size_mb": round(size_mb, 1),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        return cache_path

    def load_group_cache(self, group_key: str) -> bool:
        """Load a group KV cache entry from disk into memory.

        Returns True on success, False if not found on disk.
        """
        if group_key in self._group_cache:
            return True  # Already in memory

        cache_path = self._group_cache_dir() / f"{group_key}.pt"
        if not cache_path.exists():
            return False

        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        prefix_len = data["prefix_len"]
        cached_layers = list(zip(data["keys"], data["values"]))
        self._group_cache[group_key] = (cached_layers, prefix_len)
        return True

    def clear_group_cache(self):
        """Clear all in-memory group cache entries."""
        self._group_cache.clear()

    @staticmethod
    def compute_group_key(tool_schemas: list[str]) -> str:
        """Compute the cache key for a set of tool schemas."""
        return hashlib.sha256(
            "\n".join(sorted(tool_schemas)).encode()
        ).hexdigest()

    @property
    def group_cache_info(self) -> dict:
        """Return info about the current group cache state."""
        entries = {}
        for key, (layers, prefix_len) in self._group_cache.items():
            k0, v0 = layers[0]
            per_layer_bytes = k0.nelement() * k0.element_size() + v0.nelement() * v0.element_size()
            total_bytes = per_layer_bytes * len(layers)
            entries[key[:16]] = {
                "prefix_tokens": prefix_len,
                "num_layers": len(layers),
                "cache_size_mb": round(total_bytes / (1024 * 1024), 1),
            }

        # Also check disk index
        index_path = self._group_cache_dir() / "index.json"
        disk_entries = {}
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                disk_entries = json.load(f)

        return {
            "in_memory": entries,
            "num_in_memory": len(entries),
            "on_disk": {k[:16]: v for k, v in disk_entries.items()},
            "num_on_disk": len(disk_entries),
        }

    # ========================
    # BASELINE: FULL PREFILL
    # ========================

    @torch.no_grad()
    def generate_full_prefill(
        self,
        context_texts: list[str],
        user_query: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
    ) -> tuple[str, dict]:
        """Standard full prefill baseline — all contexts as text in prompt.

        This processes the entire prompt from scratch on every call.
        Used as the quality/latency baseline for comparison.
        """
        if system_prompt is None:
            system_prompt = self.config.system_prompt

        timings = {}

        # Build full prompt using Qwen3 chat template with tools
        full_prompt = self._build_full_prompt(context_texts, user_query, system_prompt)

        input_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False, return_tensors="pt")
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids])
        input_ids = input_ids.to(self.device)

        timings["prompt_tokens"] = input_ids.shape[1]

        # Full prefill
        t0 = time.perf_counter()
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        timings["prefill_ms"] = (time.perf_counter() - t0) * 1000

        # Autoregressive generation
        t0 = time.perf_counter()
        generated_ids = []
        next_token_logits = outputs.logits[:, -1, :]
        kv_cache = outputs.past_key_values
        current_pos = input_ids.shape[1]

        for _ in range(max_new_tokens):
            next_token = torch.argmax(next_token_logits, dim=-1)
            token_id = next_token.item()

            if token_id in self._stop_token_ids:
                break

            generated_ids.append(token_id)

            pos_ids = torch.tensor([[current_pos]], device=self.device)
            cache_pos = torch.tensor([current_pos], device=self.device)

            outputs = self.model(
                input_ids=next_token.unsqueeze(0),
                position_ids=pos_ids,
                past_key_values=kv_cache,
                cache_position=cache_pos,
                use_cache=True,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            kv_cache = outputs.past_key_values
            current_pos += 1

        timings["decode_ms"] = (time.perf_counter() - t0) * 1000
        timings["num_generated_tokens"] = len(generated_ids)

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return text, timings
