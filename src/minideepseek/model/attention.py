# Copyright (c) 2026 miniDeepSeek contributors.
# Attention math refactored from torchtitan/models/deepseek_v4/attention.py
# and attn_gym sparse gather-attention reference (BSD-3-Clause / Apache-2.0).
"""DeepSeek-V4 hybrid attention: SWA / CSA / HCA + mHC-compatible projections.

Implements V4 report §2.3: sliding-window layers plus compressed attention
(CSA, §2.3.1) and the low-rank query/output projections shared by both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn

from .compressor import Compressor, Indexer, SparseIndexerLoss, compressed_cu_seqlens
from .config import AttentionConfig
from .nn import RMSNorm, init_linear
from .rope import RoPE


@dataclass
class PackedAttentionMetadata:
    cu_seqlens: torch.Tensor
    max_seqlen: int


def _softmax_with_sink(logits: torch.Tensor, sink: torch.Tensor):
    sink = sink.to(logits.dtype)[None, :, None, None]
    logits = torch.cat((logits, sink.expand(*logits.shape[:-1], 1)), dim=-1)
    empty = torch.isneginf(logits).all(dim=-1, keepdim=True)
    safe = torch.where(empty, torch.zeros_like(logits), logits)
    probs = torch.where(empty, 0, torch.softmax(safe, dim=-1)[..., :-1])
    lse = torch.where(empty.squeeze(-1), float("-inf"), torch.logsumexp(safe, dim=-1))
    return probs, lse


def _gather_attention_reference(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sliding_window_size: int,
    *,
    scale: float,
):
    """Packed gather attention: local sliding window + selected compressed KV.

    Shapes: query ``[1, H, T, D]``, local_kv ``[1, 1, T, D]``, sparse_kv
    ``[1, 1, S, D]``, kv_indices ``[1, T, K]``.  Pure-torch reference used on
    CPU, for tests, and as the fallback when the optional fused kernel is
    unavailable.
    """
    batch, heads, tokens, dim = query.shape
    positions = torch.arange(tokens, device=query.device)
    window = min(sliding_window_size, tokens)
    local_indices = positions[:, None] - window + 1 + torch.arange(window, device=query.device)
    local_valid = (local_indices >= 0).unsqueeze(0)
    documents = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    starts = cu_seqlens[documents]
    local_valid = local_valid & (local_indices >= starts[:, None])
    local_indices = local_indices.clamp_min(0).expand(batch, -1, -1)
    sparse_starts = cu_seqlens_k[documents][None, :, None]
    sparse_ends = cu_seqlens_k[(documents + 1).clamp(max=cu_seqlens_k.shape[0] - 1)][None, :, None]
    sparse_valid = (kv_indices >= 0) & (kv_indices < sparse_ends - sparse_starts)
    sparse_indices = torch.where(sparse_valid, kv_indices, 0) + sparse_starts + tokens
    sparse_indices = torch.where(sparse_valid, sparse_indices, 0)
    indices = torch.cat((local_indices, sparse_indices), dim=-1).long()
    valid = torch.cat((local_valid, sparse_valid), dim=-1)[:, None, :, :, None]
    kv = torch.cat((local_kv, sparse_kv), dim=2)
    kv_heads = kv.shape[1]
    slots = kv[
        torch.arange(batch, device=query.device)[:, None, None, None],
        torch.arange(kv_heads, device=query.device)[None, :, None, None],
        indices[:, None],
    ]
    slots = torch.where(valid, slots, 0)
    acc = torch.promote_types(query.dtype, torch.float32)
    slots_acc = slots.to(acc)
    grouped = query.reshape(batch, kv_heads, heads // kv_heads, tokens, dim)
    grouped = grouped.permute(0, 1, 3, 2, 4).to(acc)
    logits = torch.matmul(grouped, slots_acc.transpose(-2, -1)) * scale
    logits = logits.permute(0, 1, 3, 2, 4).reshape(batch, heads, tokens, indices.shape[-1])
    logits = logits.masked_fill(~valid.squeeze(-1), float("-inf"))
    probs, lse = _softmax_with_sink(logits, attention_sink)
    probs = probs.to(query.dtype)
    grouped_probs = probs.reshape(batch, kv_heads, heads // kv_heads, tokens, indices.shape[-1]).permute(0, 1, 3, 2, 4)
    output = torch.matmul(grouped_probs.to(acc), slots_acc)
    output = output.permute(0, 1, 3, 2, 4).reshape(batch, heads, tokens, dim)
    return output.to(query.dtype), lse


_ATTN_GYM_IMPL: tuple | None | bool = None


def _load_attn_gym():
    """Lazily import attn_gym's fused gather attention.

    Returns ``(gather_attn, AuxRequest)`` or ``None``.  The package is an
    optional ``kernels`` extra; CPU-only installs and environments without it
    keep using the torch reference path.
    """
    global _ATTN_GYM_IMPL
    if _ATTN_GYM_IMPL is None:
        try:
            from attn_gym.sparse.gather_attn import AuxRequest, gather_attn

            if os.environ.get("MINIDEEPSEEK_ATTN_AUTOTUNE", "1") == "0":
                _limit_attn_gym_autotune()
            _ATTN_GYM_IMPL = (gather_attn, AuxRequest)
        except Exception:  # pragma: no cover - optional dependency
            _ATTN_GYM_IMPL = False
    return _ATTN_GYM_IMPL or None


def _limit_attn_gym_autotune() -> None:
    """Keep only the first autotune candidate per attn_gym Triton kernel.

    Triton compiles every autotune candidate on first launch, which costs tens
    of minutes for the gather-attention kernels on consumer GPUs.  The launch
    shapes are fixed for a stage, so a single candidate is enough; enabling
    ``MINIDEEPSEEK_ATTN_AUTOTUNE=0`` trades a few percent of kernel time for a
    much shorter first step.  Invalid candidates raise at launch and the
    dispatcher falls back to the torch reference.
    """
    try:
        from attn_gym._backends.triton.tune import TritonTuner
        from attn_gym.sparse.gather_attn.impl.triton import backward, forward

        for module in (forward, backward):
            for name in dir(module):
                tuner = getattr(module, name)
                if not isinstance(tuner, TritonTuner):
                    continue
                configs = list(getattr(tuner, "_configs", []))
                if len(configs) <= 1:
                    continue
                tuner._configs = configs[:1]
                tuner._candidates = {
                    id(configs[0]): tuple(sorted(configs[0].all_kwargs().items()))
                }
    except Exception:  # pragma: no cover - keep upstream autotuning on failure
        pass


def _fused_gather_attention(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sliding_window_size: int,
    *,
    scale: float,
):
    impl = _load_attn_gym()
    if impl is None:
        return None
    # RoPE/RMSNorm upcast the projections to fp32, and attn_gym's fp32 path
    # does not use tensor cores: the same kernel is ~50x slower than bf16 on a
    # 4090.  Cast at the kernel boundary and restore the caller's dtype; the
    # softmax/LSE stay fp32 inside the kernel, and the residual stream keeps
    # its original precision.  Off by default: Triton's bf16 lowering for this
    # kernel compiles for tens of minutes on a CPU-only bootstrap, so the cast
    # is opt-in until the cache is pre-warmed (`MINIDEEPSEEK_ATTN_BF16=1`).
    kernel_dtype = query.dtype
    if query.dtype == torch.float32 and os.environ.get("MINIDEEPSEEK_ATTN_BF16", "0") == "1":
        kernel_dtype = torch.bfloat16
    output_dtype = query.dtype
    query, local_kv, sparse_kv = (
        query.to(kernel_dtype),
        local_kv.to(kernel_dtype),
        sparse_kv.to(kernel_dtype),
    )
    gather_attn, aux_request = impl
    output, aux = gather_attn(
        query,
        local_kv,
        sparse_kv,
        kv_indices.to(torch.int32).contiguous(),
        attention_sink,
        sliding_window_size=int(sliding_window_size),
        cu_seqlens=cu_seqlens.to(torch.int32).contiguous(),
        cu_seqlens_k=cu_seqlens_k.to(torch.int32).contiguous(),
        scale=float(scale),
        return_aux=aux_request(lse=True),
    )
    return output.to(output_dtype), aux.lse


def gather_attention(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sliding_window_size: int,
    *,
    scale: float,
    impl: str | None = None,
):
    """Dispatch between the fused attn_gym kernel and the torch reference.

    ``MINIDEEPSEEK_ATTN_BACKEND`` selects the default: ``auto`` (fused when a
    CUDA device and attn_gym are available, otherwise reference), ``torch``
    (always reference) or ``attn_gym`` (require the fused kernel).  ``impl``
    overrides the environment for tests and A/B benchmarks.
    """
    backend = (impl or os.environ.get("MINIDEEPSEEK_ATTN_BACKEND", "auto")).lower()
    if backend not in {"auto", "torch", "attn_gym"}:
        raise ValueError(
            f"unknown attention backend {backend!r}; expected auto, torch or attn_gym"
        )
    has_slots = kv_indices.shape[-1] > 0 and query.shape[2] > 0
    if backend != "torch" and query.is_cuda and has_slots:
        try:
            fused = _fused_gather_attention(
                query,
                local_kv,
                sparse_kv,
                kv_indices,
                attention_sink,
                cu_seqlens,
                cu_seqlens_k,
                sliding_window_size,
                scale=scale,
            )
        except Exception as error:  # pragma: no cover - depends on the kernel
            if backend == "attn_gym":
                raise RuntimeError(f"attn_gym gather attention failed: {error}") from error
            fused = None
        if fused is not None:
            return fused
    elif backend == "attn_gym" and not query.is_cuda:
        raise RuntimeError("attn_gym gather attention requires CUDA tensors")
    return _gather_attention_reference(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        cu_seqlens,
        cu_seqlens_k,
        sliding_window_size,
        scale=scale,
    )


class GroupedOutProjection(nn.Module):
    """Block-diagonal ``wo_a``: each head group projects to o_lora_rank."""

    def __init__(self, n_groups: int, o_lora_rank: int, per_group_in: int):
        super().__init__()
        self.n_groups = n_groups
        self.o_lora_rank = o_lora_rank
        self.weight = nn.Parameter(torch.empty(n_groups * o_lora_rank, per_group_in))

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[0]
        groups = x.view(tokens, self.n_groups, -1)
        weight = self.weight.view(self.n_groups, self.o_lora_rank, -1)
        return torch.einsum("tgd,god->tgo", groups, weight).reshape(
            tokens, self.n_groups * self.o_lora_rank
        )


class DeepSeekV4Attention(nn.Module):
    def __init__(self, config: AttentionConfig, *, max_context_length: int):
        super().__init__()
        c = config
        self.dim = c.dim
        self.n_heads = c.n_heads
        self.head_dim = c.head_dim
        self.rope_head_dim = c.rope_head_dim
        self.nope_dim = c.nope_dim
        self.q_lora_rank = c.q_lora_rank
        self.o_lora_rank = c.o_lora_rank
        self.n_groups = c.n_groups
        self.compress_ratio = c.compress_ratio
        self.window_size = c.window_size
        self.index_topk = c.index_topk
        self.norm_eps = c.norm_eps
        self.softmax_scale = c.head_dim**-0.5
        self.layer_id = c.layer_id
        self.n_layers = c.n_layers
        self.use_indexer_aux = c.use_indexer_aux and c.compress_ratio == 4
        self.indexer_loss_coeff = c.indexer_loss_coeff
        self.indexer_aux = (
            SparseIndexerLoss(
                num_heads=c.n_heads,
                compress_ratio=c.compress_ratio,
                coeff=c.indexer_loss_coeff,
                softmax_scale=c.head_dim**-0.5,
            )
            if self.use_indexer_aux
            else None
        )
        self.last_indexer_loss: torch.Tensor | None = None

        self.wq_a = nn.Linear(c.dim, c.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(c.q_lora_rank, c.norm_eps)
        self.wq_b = nn.Linear(c.q_lora_rank, c.n_heads * c.head_dim, bias=False)
        self.wkv = nn.Linear(c.dim, c.head_dim, bias=False)
        self.kv_norm = RMSNorm(c.head_dim, c.norm_eps)
        self.wo_a = GroupedOutProjection(c.n_groups, c.o_lora_rank, c.per_group_in)
        self.wo_b = nn.Linear(c.per_group_out, c.dim, bias=False)
        self.attn_sink = nn.Linear(1, c.n_heads, bias=False)

        # Compressed layers rotate with the compressor's YaRN RoPE end to end,
        # matching the reference implementation.
        main_rope_yarn = c.compress_rope_yarn if c.compress_ratio > 1 else c.rope_yarn
        main_rope_theta = c.compress_rope_theta if c.compress_ratio > 1 else c.rope_theta
        self.rope = RoPE(
            c.rope_head_dim,
            max_context_length,
            theta=main_rope_theta,
            yarn=main_rope_yarn,
            factor=c.rope_factor,
            original_seq_len=c.original_seq_len,
        )
        self.compress_rope = RoPE(
            c.rope_head_dim,
            max_context_length,
            theta=c.compress_rope_theta,
            yarn=c.compress_rope_yarn,
            factor=c.rope_factor,
            original_seq_len=c.original_seq_len,
        )
        compressor = (
            Compressor(
                c.dim,
                c.head_dim,
                c.rope_head_dim,
                c.compress_ratio,
                overlap=c.compress_ratio == 4,
                rope=self.compress_rope,
                norm_eps=c.norm_eps,
            )
            if c.has_compressor
            else None
        )
        # Match the upstream attribute names so state dicts are interchangeable.
        if c.compress_ratio == 4:
            self.compressor = compressor
        elif c.has_compressor:
            self.compressor_128 = compressor
        self.indexer = (
            Indexer(
                c.dim,
                c.q_lora_rank,
                c.index_n_heads,
                c.index_head_dim,
                c.compress_ratio,
                rope=self.compress_rope,
                norm_eps=c.norm_eps,
            )
            if c.has_indexer
            else None
        )

    def init_weights(self) -> None:
        for module in (self.wq_a, self.wq_b, self.wkv, self.wo_b):
            init_linear(module)
        self.wo_a.init_weights()
        init_linear(self.attn_sink)
        nn.init.ones_(self.q_norm.weight)
        nn.init.ones_(self.kv_norm.weight)
        for module in (getattr(self, "compressor", None), getattr(self, "compressor_128", None)):
            if module is not None:
                module.init_weights()
        if self.indexer is not None:
            self.indexer.init_weights()

    def _hca_topk(self, tokens: int, cu_seqlens: torch.Tensor, n_candidates: int, device):
        positions = torch.arange(tokens, device=device, dtype=torch.int32)
        doc = torch.searchsorted(cu_seqlens[1:], positions, right=True, out_int32=True)
        local = positions - cu_seqlens[doc]
        limit = (local + 1) // self.compress_ratio
        width = min(n_candidates, int(limit.max().item()) if tokens else 0)
        if width <= 0:
            return torch.empty(tokens, 0, dtype=torch.int32, device=device)
        cmp = torch.arange(width, device=device, dtype=torch.int32)
        return torch.where(cmp[None, :] < limit[:, None], cmp[None, :], -1)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: PackedAttentionMetadata,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        del input_ids, padding_mask, aux_loss_denominator
        tokens = x.shape[0]
        cu = attention_metadata.cu_seqlens
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(tokens, self.n_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.nope_dim, self.rope_head_dim], dim=-1)
        kv = self.kv_norm(self.wkv(x))
        kv_nope, kv_rope = torch.split(kv, [self.nope_dim, self.rope_head_dim], dim=-1)
        q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(1), positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)

        cmp_k = None
        topk = None
        compressor = getattr(self, "compressor", None) or getattr(self, "compressor_128", None)
        if compressor is not None:
            cmp_k = compressor(x, positions, cu)
        if self.indexer is not None and cmp_k is not None:
            idx_q, idx_k, idx_w = self.indexer(
                x.detach(), qr.detach(), positions=positions, cu_seqlens=cu
            )
            topk, topk_scores = Indexer.select(
                idx_q,
                idx_k,
                idx_w,
                topk=min(self.index_topk, max(idx_k.shape[0], 1)),
                ratio=self.compress_ratio,
                cu_seqlens=cu,
                return_scores=self.training and self.indexer_aux is not None,
            )
        else:
            topk_scores = None
            if cmp_k is not None:
                topk = self._hca_topk(tokens, cu, cmp_k.shape[0], x.device)
            else:
                topk = torch.empty(tokens, 0, dtype=torch.int32, device=x.device)

        if cmp_k is None:
            sparse_kv = q.new_zeros(1, 1, 0, self.head_dim)
            cu_k = torch.zeros_like(cu)
        else:
            sparse_kv = cmp_k.unsqueeze(0).unsqueeze(0)
            cu_k = compressed_cu_seqlens(cu, self.compress_ratio)
        out, lse = gather_attention(
            q.transpose(0, 1).unsqueeze(0),
            kv.unsqueeze(0).unsqueeze(0),
            sparse_kv,
            topk.unsqueeze(0),
            self.attn_sink.weight.squeeze(-1),
            cu,
            cu_k,
            self.window_size,
            scale=self.softmax_scale,
        )
        out = out.squeeze(0).transpose(0, 1)
        self.last_indexer_loss = None
        if (
            self.training
            and self.indexer_aux is not None
            and topk_scores is not None
            and cmp_k is not None
            and topk.numel() > 0
        ):
            # lse is [1, H, T]; the auxiliary loss expects [T, H].
            self.last_indexer_loss = self.indexer_aux(
                q.detach(),
                cmp_k.detach(),
                topk,
                topk_scores,
                lse.squeeze(0).transpose(0, 1),
                cu,
                denominator=tokens,
            )
        o_nope, o_rope = torch.split(out, [self.nope_dim, self.rope_head_dim], dim=-1)
        o_rope = self.rope(o_rope, positions=positions, inverse=True)
        out = torch.cat([o_nope, o_rope], dim=-1)
        out = self.wo_b(self.wo_a(out))
        return out
