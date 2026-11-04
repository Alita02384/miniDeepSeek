# Copyright (c) 2026 miniDeepSeek contributors.
# Refactored from torchtitan/models/common/rope.py (BSD-3-Clause, Meta).
"""Rotary position embedding with optional YaRN, in real arithmetic."""

from __future__ import annotations

import math

import torch
from torch import nn


def yarn_inv_freq(
    dim: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    original_seq_len: int,
) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))

    def correction(num_rotations: float) -> float:
        return (dim * math.log(original_seq_len / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction(beta_fast)), 0)
    high = min(math.ceil(correction(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    return inv_freq / factor * ramp + inv_freq * (1 - ramp)


class RoPE(nn.Module):
    """Adjacent-pair rotary embedding.

    ``positions`` is a ``[T]`` tensor.  Query/key shapes are ``[T, heads, dim]``.
    The cache stores cos/sin as float32 and is registered as a buffer so it
    follows ``.to(device)`` and checkpoints.
    """

    def __init__(
        self,
        dim: int,
        max_context_length: int,
        *,
        theta: float = 10000.0,
        yarn: bool = False,
        factor: float = 4.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        original_seq_len: int = 65536,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")
        self.dim = dim
        self.max_context_length = max_context_length
        if yarn and factor > 1.0:
            inv_freq = yarn_inv_freq(
                dim, theta, factor, beta_fast, beta_slow, original_seq_len
            )
        else:
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        positions = torch.arange(max_context_length, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq.float())
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        *,
        inverse: bool = False,
    ):
        tokens = query.shape[0]
        if positions is None:
            positions = torch.arange(tokens, device=query.device)
        cos = self.cos_cached[positions].to(query.dtype)
        sin = self.sin_cached[positions].to(query.dtype)
        if inverse:
            sin = -sin
        # [T, 1, D/2] broadcasts over heads.
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query_out = _rotate_pairs(query, cos, sin)
        if key is None:
            return query_out
        return query_out, _rotate_pairs(key, cos, sin)


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    pairs = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)
