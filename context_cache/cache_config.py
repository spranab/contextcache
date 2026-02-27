"""Configuration for ContextCache."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3-8B"
    adapter: str = "auto"  # "auto", "qwen", "llama", "mistral" — auto-detects from model_name
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    max_seq_length: int = 4096


@dataclass
class CacheStorageConfig:
    cache_dir: str = "cache/context_kv"
    num_dummy_bos: int = 4  # LegoLink-0 dummy tokens
    preload_to_gpu: bool = True
    max_gpu_cache_mb: int = 4000


@dataclass
class RoPEConfig:
    theta: float = 1_000_000.0  # Qwen3 rope_theta
    max_position: int = 8192


@dataclass
class EvalConfig:
    max_examples: int = 200
    max_new_tokens: int = 256
    catalogs: list[str] = field(default_factory=lambda: [
        "data/catalogs/catalog_100.json",
        "data/catalogs/catalog_unseen.json",
    ])
    test_splits: list[str] = field(default_factory=lambda: [
        "data/processed/test_seen_gisting.jsonl",
        "data/processed/test_held_out_gisting.jsonl",
        "data/processed/test_unseen_gisting.jsonl",
    ])


@dataclass
class GenerationConfig:
    force_tool_call_prefix: bool = True
    stop_on_tool_call_end: bool = True
    max_new_tokens: int = 256


@dataclass
class ContextCacheConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheStorageConfig = field(default_factory=CacheStorageConfig)
    rope: RoPEConfig = field(default_factory=RoPEConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    system_prompt: str = "You are a helpful assistant with access to the following tools. Use them when appropriate to help the user."

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ContextCacheConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        model_cfg = ModelConfig(**data.get("model", {}))
        cache_cfg = CacheStorageConfig(**data.get("cache", {}))
        rope_cfg = RoPEConfig(**data.get("rope", {}))
        eval_cfg = EvalConfig(**data.get("eval", {}))
        gen_cfg = GenerationConfig(**data.get("generation", {}))
        system_prompt = data.get("system_prompt", cls.system_prompt)

        return cls(
            model=model_cfg,
            cache=cache_cfg,
            rope=rope_cfg,
            eval=eval_cfg,
            generation=gen_cfg,
            system_prompt=system_prompt,
        )
