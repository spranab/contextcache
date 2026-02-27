"""Model adapters for ContextCache — abstract away model-specific details.

The group caching mechanism is model-agnostic:
  1. Forward pass prefix+tools → store KV cache
  2. On cache hit → restore KV, forward only user query

What differs between models:
  - How to format the prompt (chat template, tool format)
  - How to access KV cache layers
  - Stop tokens for generation
  - RoPE configuration (only needed for NoPE path, not group caching)

Usage:
  adapter = QwenAdapter(model, tokenizer, config)  # or LlamaAdapter, etc.
  prompt = adapter.build_full_prompt(tool_schemas, user_query, system_prompt)
  prefix, suffix = adapter.build_prompt_parts(tool_schemas, user_query, system_prompt)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelAdapter(ABC):
    """Base class for model-specific prompt formatting and layer access."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def build_full_prompt(
        self,
        tool_schemas: list[str],
        user_query: str,
        system_prompt: str,
    ) -> str:
        """Build the complete prompt with tools, user query, and generation prompt.

        Args:
            tool_schemas: List of tool schema JSON strings.
            user_query: The user's question.
            system_prompt: System instructions.

        Returns:
            Complete prompt string ready for tokenization.
        """
        ...

    @abstractmethod
    def build_prompt_parts(
        self,
        tool_schemas: list[str],
        user_query: str,
        system_prompt: str,
    ) -> tuple[str, str]:
        """Split prompt into prefix (system+tools) and suffix (user query+gen prompt).

        The prefix contains everything that can be cached (system prompt + tool
        definitions). The suffix is per-request (user query + generation prompt).

        Returns:
            (prefix_text, suffix_text)
        """
        ...

    @abstractmethod
    def get_stop_token_ids(self) -> set[int]:
        """Return the set of token IDs that should stop generation."""
        ...

    def get_layer_devices(self) -> list:
        """Get the device each transformer layer resides on."""
        import torch
        devices = []
        for layer in self._get_layers():
            param = next(layer.parameters())
            devices.append(param.device)
        return devices

    @abstractmethod
    def _get_layers(self):
        """Return the list of transformer layers."""
        ...

    @property
    def num_layers(self) -> int:
        return len(self._get_layers())

    @property
    def rope_theta(self) -> float:
        """Return the RoPE base frequency from model config."""
        cfg = self.model.config
        return getattr(cfg, "rope_theta", 10000.0)

    @property
    def num_kv_heads(self) -> int:
        return self.model.config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        cfg = self.model.config
        return getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )

    def monkey_patch_rope_capture(self, capture_state: dict):
        """Install a monkey-patch to capture pre-RoPE keys (for NoPE path).

        Args:
            capture_state: Dict with keys 'enabled' (bool), 'counter' (int),
                          'buffer' (dict[int, Tensor]). The adapter will
                          read/write these during forward passes.

        Not all adapters need to implement this — it's only for the NoPE path.
        The group caching path doesn't need it.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support NoPE capture. "
            "Use group caching instead."
        )


class QwenAdapter(ModelAdapter):
    """Adapter for Qwen3 models (Qwen3-8B, Qwen3-4B, etc.)."""

    def build_full_prompt(self, tool_schemas, user_query, system_prompt):
        tools = self._parse_tools(tool_schemas)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def build_prompt_parts(self, tool_schemas, user_query, system_prompt):
        full_prompt = self.build_full_prompt(tool_schemas, user_query, system_prompt)

        # Qwen3 wraps tools in <tools>...</tools> XML blocks
        tools_start = full_prompt.find("<tools>\n")
        tools_end = full_prompt.rfind("</tools>")

        if tools_start < 0 or tools_end < 0 or tools_end <= tools_start:
            return full_prompt, ""

        prefix = full_prompt[:tools_start + len("<tools>\n")]
        # Extract tools text for cache key
        tools_text = full_prompt[tools_start + len("<tools>\n"):tools_end]
        # Prefix includes everything through tools
        prefix = full_prompt[:tools_end + len("</tools>")]
        suffix = full_prompt[tools_end + len("</tools>"):]

        return prefix, suffix

    def get_stop_token_ids(self):
        ids = {self.tokenizer.eos_token_id}
        endoftext = self.tokenizer.encode("<|endoftext|>", add_special_tokens=False)
        if endoftext:
            ids.add(endoftext[0])
        return ids

    def _get_layers(self):
        return self.model.model.layers

    def monkey_patch_rope_capture(self, capture_state):
        import transformers.models.qwen3.modeling_qwen3 as qwen3_module
        original = qwen3_module.apply_rotary_pos_emb

        def patched(*args, **kwargs):
            if capture_state["enabled"]:
                k = args[1]
                layer_idx = capture_state["counter"]
                capture_state["buffer"][layer_idx] = k.detach().clone()
                capture_state["counter"] += 1
            return original(*args, **kwargs)

        qwen3_module.apply_rotary_pos_emb = patched

    @staticmethod
    def _parse_tools(tool_schemas: list[str]) -> list[dict]:
        tools = []
        for s in tool_schemas:
            try:
                tools.append(json.loads(s))
            except json.JSONDecodeError:
                continue
        return tools


class LlamaAdapter(ModelAdapter):
    """Adapter for Llama 3.x models with tool calling support.

    Llama 3.1+ supports tool calling via a specific chat template format.
    Tools are passed as a list of dicts with function definitions.
    """

    def build_full_prompt(self, tool_schemas, user_query, system_prompt):
        tools = self._parse_tools(tool_schemas)

        # Llama 3.1+ uses the same HF chat template API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        # Try using built-in tool support first
        try:
            return self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: embed tools in system prompt
            return self._manual_prompt(tools, user_query, system_prompt)

    def build_prompt_parts(self, tool_schemas, user_query, system_prompt):
        tools = self._parse_tools(tool_schemas)

        # Build system+tools as prefix
        sys_with_tools = system_prompt + "\n\nAvailable tools:\n"
        for tool in tools:
            sys_with_tools += json.dumps(tool, indent=2) + "\n"

        prefix_messages = [{"role": "system", "content": sys_with_tools}]
        suffix_messages = [{"role": "user", "content": user_query}]

        try:
            prefix = self.tokenizer.apply_chat_template(
                prefix_messages, tokenize=False, add_generation_prompt=False,
            )
            # For suffix, we need just the user turn + gen prompt
            full = self.tokenizer.apply_chat_template(
                prefix_messages + suffix_messages, tokenize=False,
                add_generation_prompt=True,
            )
            suffix = full[len(prefix):]
        except Exception:
            full = self._manual_prompt(tools, user_query, system_prompt)
            # Simple split: everything before user query is prefix
            idx = full.rfind(user_query)
            if idx > 0:
                prefix = full[:idx]
                suffix = full[idx:]
            else:
                prefix = full
                suffix = ""

        return prefix, suffix

    def get_stop_token_ids(self):
        ids = {self.tokenizer.eos_token_id}
        # Llama 3 uses <|eot_id|> as end-of-turn
        for special in ["<|eot_id|>", "<|end_of_text|>"]:
            encoded = self.tokenizer.encode(special, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
        return ids

    def _get_layers(self):
        return self.model.model.layers

    def _manual_prompt(self, tools, user_query, system_prompt):
        """Fallback prompt format when chat template doesn't support tools."""
        tools_str = "\n".join(json.dumps(t, indent=2) for t in tools)
        content = f"{system_prompt}\n\nAvailable tools:\n{tools_str}"
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": user_query},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    @staticmethod
    def _parse_tools(tool_schemas):
        tools = []
        for s in tool_schemas:
            try:
                tools.append(json.loads(s))
            except json.JSONDecodeError:
                continue
        return tools


class MistralAdapter(ModelAdapter):
    """Adapter for Mistral/Mixtral models."""

    def build_full_prompt(self, tool_schemas, user_query, system_prompt):
        tools = self._parse_tools(tool_schemas)
        tools_str = "\n".join(json.dumps(t) for t in tools)
        content = f"{system_prompt}\n\n[AVAILABLE_TOOLS]{tools_str}[/AVAILABLE_TOOLS]"
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": user_query},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def build_prompt_parts(self, tool_schemas, user_query, system_prompt):
        full = self.build_full_prompt(tool_schemas, user_query, system_prompt)
        # Split at the user message boundary
        marker = "[/AVAILABLE_TOOLS]"
        idx = full.find(marker)
        if idx > 0:
            prefix = full[:idx + len(marker)]
            suffix = full[idx + len(marker):]
        else:
            prefix = full
            suffix = ""
        return prefix, suffix

    def get_stop_token_ids(self):
        ids = {self.tokenizer.eos_token_id}
        for special in ["</s>", "[/INST]"]:
            encoded = self.tokenizer.encode(special, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
        return ids

    def _get_layers(self):
        return self.model.model.layers

    @staticmethod
    def _parse_tools(tool_schemas):
        tools = []
        for s in tool_schemas:
            try:
                tools.append(json.loads(s))
            except json.JSONDecodeError:
                continue
        return tools


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {
    "qwen": QwenAdapter,
    "qwen3": QwenAdapter,
    "llama": LlamaAdapter,
    "mistral": MistralAdapter,
    "mixtral": MistralAdapter,
}


def get_adapter(model_name: str, model, tokenizer) -> ModelAdapter:
    """Auto-detect and return the right adapter for a model.

    Checks the model name against known patterns. Falls back to Llama
    adapter (most common HF chat template format) if unknown.
    """
    name_lower = model_name.lower()

    for key, adapter_cls in ADAPTER_REGISTRY.items():
        if key in name_lower:
            return adapter_cls(model, tokenizer)

    # Default: try Llama adapter (most standard HF chat template)
    return LlamaAdapter(model, tokenizer)
