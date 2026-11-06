# Copyright (c) 2026 miniDeepSeek contributors.
# Refactored from torchtitan/models/deepseek_v4/mhc.py (BSD-3-Clause, Meta).
"""Multi-branch HC mixing (mHC): pre/post mixing and Sinkhorn normalization.

Implements V4 report §2.2; the Sinkhorn iteration follows the official
reference implementation (20 iterations, log-space).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HcSplitSinkhorn(nn.Module):
    def __init__(self, hc_mult: int, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def forward(self, mixes, hc_scale, hc_base):
        hc = self.hc_mult
        pre, post, comb = mixes.split([hc, hc, hc * hc], dim=-1)
        comb = comb.unflatten(-1, (hc, hc))
        pre = torch.sigmoid(
            pre * hc_scale[0] + hc_base[:hc].view(*([1] * (pre.ndim - 1)), hc)
        ) + self.eps
        post = 2 * torch.sigmoid(
            post * hc_scale[1] + hc_base[hc : 2 * hc].view(*([1] * (post.ndim - 1)), hc)
        )
        comb = comb * hc_scale[2] + hc_base[2 * hc :].view(*([1] * (comb.ndim - 2)), hc, hc)
        row_max = comb.max(dim=-1, keepdim=True).values
        comb = torch.exp(comb - row_max)
        comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        return pre, post, comb


class HcPre(nn.Module):
    def __init__(
        self,
        dim: int,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        mix = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_fn = nn.Parameter(torch.empty(mix, hc_dim))
        self.hc_base = nn.Parameter(torch.empty(mix))
        self.hc_scale = nn.Parameter(torch.empty(3))
        self.sinkhorn = HcSplitSinkhorn(hc_mult, sinkhorn_iters, eps)

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.hc_fn, std=0.02)
        nn.init.trunc_normal_(self.hc_base, std=0.02)
        nn.init.trunc_normal_(self.hc_scale, std=0.02)

    def forward(self, x):
        shape, dtype = x.size(), x.dtype
        flat = x.flatten(-2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_fn.float()) * rsqrt
        pre, post, comb = self.sinkhorn(mixes, self.hc_scale.float(), self.hc_base.float())
        y = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=-2)
        return y.to(dtype), post, comb


class HcPost(nn.Module):
    def forward(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1
        )
        return y.type_as(x)


class HcHead(nn.Module):
    def __init__(self, dim: int, hc_mult: int = 4, norm_eps: float = 1e-6, eps: float = 1e-6):
        super().__init__()
        hc_dim = hc_mult * dim
        self.norm_eps = norm_eps
        self.eps = eps
        self.hc_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
        self.hc_base = nn.Parameter(torch.empty(hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.hc_fn, std=0.02)
        nn.init.trunc_normal_(self.hc_base, std=0.02)
        nn.init.trunc_normal_(self.hc_scale, std=0.02)

    def forward(self, x):
        shape, dtype = x.size(), x.dtype
        flat = x.flatten(-2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale + self.hc_base) + self.eps
        y = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=-2)
        return y.to(dtype)
