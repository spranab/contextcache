"""Deferred RoPE utilities for position-independent KV cache composition.

Implements the core math for:
1. Building cos/sin tables matching Qwen3's RoPE implementation
2. Applying RoPE to pre-RoPE (NoPE) key states at arbitrary positions
3. Reversing RoPE for verification purposes

Reference: Qwen3 uses standard RoPE with theta=1,000,000, applied after QK-norm
but before KV cache update. We intercept pre-RoPE keys and defer rotation to
link time, enabling position-independent caching.
"""

from __future__ import annotations

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input.
    Matches transformers.models.qwen3.modeling_qwen3.rotate_half exactly.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    rope_theta: float = 1_000_000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Pre-compute cos/sin tables for RoPE, matching Qwen3's implementation.

    Args:
        max_seq_len: Maximum sequence length to pre-compute for.
        head_dim: Dimension of each attention head (128 for Qwen3-8B).
        rope_theta: RoPE base frequency (1,000,000 for Qwen3).
        device: Target device.
        dtype: Output dtype (float32 for accuracy, cast later if needed).

    Returns:
        (cos, sin) each of shape (max_seq_len, head_dim).
        These match what Qwen3RotaryEmbedding.forward() produces for
        position_ids = [0, 1, ..., max_seq_len-1].
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    # Shape: (max_seq_len,)
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    # Shape: (max_seq_len, head_dim // 2)
    freqs = torch.outer(positions, inv_freq)
    # Shape: (max_seq_len, head_dim) — duplicate for both halves
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def apply_rope(
    keys: Tensor,
    positions: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Apply RoPE to pre-RoPE (NoPE) keys at given absolute positions.

    This is the core "deferred RoPE" operation: cached NoPE keys are rotated
    to their correct absolute positions at link time.

    Args:
        keys: Pre-RoPE key states, shape (batch, num_kv_heads, seq_len, head_dim).
        positions: Absolute position IDs for each key, shape (seq_len,).
        cos: Cosine table from build_rope_cache, shape (max_seq_len, head_dim).
        sin: Sine table from build_rope_cache, shape (max_seq_len, head_dim).

    Returns:
        Rotated key states, same shape as input.
    """
    # Gather cos/sin for the given positions: (seq_len, head_dim)
    cos_pos = cos[positions]
    sin_pos = sin[positions]
    # Unsqueeze for broadcasting: (1, 1, seq_len, head_dim)
    cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)
    sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)
    # Apply rotation (matches Qwen3's apply_rotary_pos_emb for keys)
    return (keys * cos_pos) + (rotate_half(keys) * sin_pos)


def apply_rope_to_query(
    queries: Tensor,
    positions: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Apply RoPE to query states at given positions.

    Same math as apply_rope but kept separate for clarity.
    Needed when we compute the user query's Q states fresh.

    Args:
        queries: Query states, shape (batch, num_heads, seq_len, head_dim).
        positions: Absolute position IDs, shape (seq_len,).
        cos: Cosine table, shape (max_seq_len, head_dim).
        sin: Sine table, shape (max_seq_len, head_dim).

    Returns:
        Rotated query states, same shape as input.
    """
    cos_pos = cos[positions].unsqueeze(0).unsqueeze(0)
    sin_pos = sin[positions].unsqueeze(0).unsqueeze(0)
    return (queries * cos_pos) + (rotate_half(queries) * sin_pos)


def reverse_rope(
    keys: Tensor,
    positions: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Reverse RoPE rotation — recover pre-RoPE keys from post-RoPE keys.

    Since RoPE is an orthogonal rotation, its inverse is the rotation with
    negated angle: R(-theta) = R(theta)^{-1}.

    Used for verification: forward with RoPE → reverse → should equal original.

    Args:
        keys: Post-RoPE key states, shape (batch, num_kv_heads, seq_len, head_dim).
        positions: Absolute position IDs, shape (seq_len,).
        cos: Cosine table, shape (max_seq_len, head_dim).
        sin: Sine table, shape (max_seq_len, head_dim).

    Returns:
        Pre-RoPE key states, same shape as input.
    """
    cos_pos = cos[positions].unsqueeze(0).unsqueeze(0)
    sin_pos = sin[positions].unsqueeze(0).unsqueeze(0)
    # Negate sin for inverse rotation
    return (keys * cos_pos) + (rotate_half(keys) * (-sin_pos))
