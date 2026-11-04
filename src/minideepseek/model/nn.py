# Copyright (c) 2026 miniDeepSeek contributors.
"""Small shared building blocks for the V4-style model."""

from __future__ import annotations

import math
from functools import cache

import torch
import torch.nn.functional as F
from torch import nn


def depth_scaled_std(base_std: float, layer_id: int) -> float:
    return base_std / math.sqrt(2.0 * (layer_id + 1))


def init_linear(module: nn.Linear, std: float = 0.02) -> None:
    nn.init.trunc_normal_(module.weight, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_output_linear(module: nn.Linear, dim: int) -> None:
    std = dim**-0.5
    nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class RMSNorm(nn.RMSNorm):
    """``nn.RMSNorm`` with the same constructor shape used across the project."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim, eps=eps)


class LimitedSwiGLU(nn.Module):
    """SwiGLU with the V4 clamp: gate <= limit, |up| <= limit.

    ``limit=None`` disables clamping (used by the upstream debug recipe).  The
    activation is an ``nn.Module`` so the clamp monitor can instrument it.
    """

    def __init__(self, limit: float | None = None):
        super().__init__()
        self.limit = limit

    def _activation_fn(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        return F.silu(gate) * up

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return self._activation_fn(gate, up)


def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float | None) -> torch.Tensor:
    """Functional form kept for tests and ad-hoc use."""
    return LimitedSwiGLU(limit)(gate, up)


@cache
def hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Sylvester Hadamard matrix used by the indexer rotation."""
    if dim & (dim - 1) != 0:
        raise ValueError("Hadamard dim must be a power of two")
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < dim:
        top = torch.cat([matrix, matrix], dim=1)
        bottom = torch.cat([matrix, -matrix], dim=1)
        matrix = torch.cat([top, bottom], dim=0)
    return matrix
