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

from ..paths import configure_compile_cache
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
    query_chunk: int = 128,
):
    """Packed gather attention: local sliding window + selected compressed KV.

    Shapes: query ``[1, H, T, D]``, local_kv ``[1, 1, T, D]``, sparse_kv
    ``[1, 1, S, D]``, kv_indices ``[1, T, K]``.  Pure-torch reference used on
    CPU, for tests, and as the fallback when the optional fused kernel is
    unavailable.

    Queries are processed in chunks.  Materialising ``[H, T, K, D]`` for a long
    prefill is hundreds of MiB per layer; chunking keeps the working set small
    without changing the arithmetic (each query still sees exactly the same
    keys).
    """
    batch, heads, tokens, dim = query.shape
    device = query.device
    window = min(sliding_window_size, tokens)
    all_positions = torch.arange(tokens, device=device)
    local_indices = all_positions[:, None] - window + 1 + torch.arange(window, device=device)
    local_valid = (local_indices >= 0).unsqueeze(0)
    documents = torch.searchsorted(cu_seqlens[1:], all_positions, right=True)
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
    acc = torch.promote_types(query.dtype, torch.float32)
    outputs = []
    lses = []
    chunk = max(int(query_chunk), 1)
    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        query_chunk_tensor = query[:, :, start:stop]
        chunk_indices = indices[:, start:stop]
        chunk_valid = valid[:, :, start:stop]
        slots = kv[
            torch.arange(batch, device=device)[:, None, None, None],
            torch.arange(kv_heads, device=device)[None, :, None, None],
            chunk_indices[:, None],
        ]
        slots = torch.where(chunk_valid, slots, 0)
        slots_acc = slots.to(acc)
        grouped = query_chunk_tensor.reshape(
            batch, kv_heads, heads // kv_heads, stop - start, dim
        )
        grouped = grouped.permute(0, 1, 3, 2, 4).to(acc)
        logits = torch.matmul(grouped, slots_acc.transpose(-2, -1)) * scale
        logits = logits.permute(0, 1, 3, 2, 4).reshape(
            batch, heads, stop - start, indices.shape[-1]
        )
        logits = logits.masked_fill(~chunk_valid.squeeze(-1), float("-inf"))
        probs, lse = _softmax_with_sink(logits, attention_sink)
        probs = probs.to(query.dtype)
        grouped_probs = probs.reshape(
            batch, kv_heads, heads // kv_heads, stop - start, indices.shape[-1]
        ).permute(0, 1, 3, 2, 4)
        output = torch.matmul(grouped_probs.to(acc), slots_acc)
        output = output.permute(0, 1, 3, 2, 4).reshape(batch, heads, stop - start, dim)
        outputs.append(output.to(query.dtype))
        lses.append(lse)
    return torch.cat(outputs, dim=2), torch.cat(lses, dim=2)


_ATTN_GYM_IMPL: tuple | None | bool = None


def _load_attn_gym():
    """Lazily import attn_gym's fused gather attention.

    Returns ``(gather_attn, AuxRequest)`` or ``None``.  The package is an
    optional ``kernels`` extra; CPU-only installs and environments without it
    keep using the torch reference path.

    Nothing here trims attn_gym's autotune candidates.  Doing so used to look
    like a way to avoid the first-step compile, but the candidate it kept was a
    Hopper/Blackwell tile that compiles pathologically on SM89 (over 40 minutes
    without producing a binary).  The cold cost is paid once instead, through
    ``scripts/prewarm_attn.py`` and the persistent cache configured by
    :func:`minideepseek.paths.configure_compile_cache`.
    """
    global _ATTN_GYM_IMPL
    if _ATTN_GYM_IMPL is None:
        try:
            # attn_gym imports Triton, which reads TRITON_CACHE_DIR at import
            # time; keep the compiled kernels in the persistent repo cache so a
            # restarted box does not recompile them.
            configure_compile_cache()
            from attn_gym.sparse.gather_attn import AuxRequest, gather_attn

            _ATTN_GYM_IMPL = (gather_attn, AuxRequest)
        except Exception:  # pragma: no cover - optional dependency
            _ATTN_GYM_IMPL = False
    return _ATTN_GYM_IMPL or None


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
    # its original precision.  Off by default because the bf16 lowering adds a
    # second set of kernel variants to compile: turn it on together with
    # ``MINIDEEPSEEK_ATTN_BF16=1 python scripts/prewarm_attn.py``, never on its
    # own on a cold cache.
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
        self.hca_topk_max = c.hca_topk_max
        self.max_doc_tokens = c.max_doc_tokens
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

    def _hca_topk(self, tokens: int, cu_seqlens: torch.Tensor, device):
        positions = torch.arange(tokens, device=device, dtype=torch.int32)
        doc = torch.searchsorted(cu_seqlens[1:], positions, right=True, out_int32=True)
        local = positions - cu_seqlens[doc]
        limit = (local + 1) // self.compress_ratio
        # Static bound, not ``limit.max()``: HCA's top-k is the kernel's TOPK
        # constexpr, so deriving it from the batch makes every packing layout a
        # new Triton specialization (minutes of compile *per layout*, repeatedly,
        # mid-training) and costs a host sync per layer.  Documents shorter than
        # the bound are masked by ``cmp < limit`` below.
        width = -(-int(self.max_doc_tokens) // self.compress_ratio)
        if self.hca_topk_max is not None:
            # Kernel-side top-k is a constexpr and statically unrolled, so an
            # uncapped 8K width (256) costs ~1 h to compile; keep the most
            # recent entries instead (only the 8K stage is affected).
            width = min(width, int(self.hca_topk_max))
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
        cache=None,
        slot: int | None = None,
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
        idx_k = None
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
                topk = self._hca_topk(tokens, cu, x.device)
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
        if cache is not None:
            if slot is None:
                raise ValueError("prefill cache requires a slot id")
            comp_state = None
            idx_state = None
            compressor = getattr(self, "compressor", None) or getattr(
                self, "compressor_128", None
            )
            if compressor is not None:
                comp_state = compressor.capture_state(x)
            if self.indexer is not None:
                idx_state = self.indexer.compressor.capture_state(x)
            cache.store_prefill(
                self.layer_id,
                int(slot),
                kv=kv,
                cmp_kv=cmp_k.detach() if cmp_k is not None else None,
                idx_kv=idx_k.detach() if idx_k is not None else None,
                comp_state=comp_state,
                idx_state=idx_state,
                num_tokens=tokens,
            )
        return out

    # -- incremental decode ---------------------------------------------------
    def _select_sparse(
        self,
        layer,
        slots: torch.Tensor,
        positions: torch.Tensor,
        idx_q: torch.Tensor | None,
        idx_w: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[B, K]`` local compressed indices and a validity mask.

        ``K`` is a static per-layer bound (``index_topk`` for CSA, the whole
        compressed axis for HCA), so neither the shape nor a host-side
        ``n_comp.max()`` changes from step to step: decode can be captured by a
        CUDA graph.  Slots with fewer entries are masked, and ``-1`` marks
        padding.
        """
        device = positions.device
        batch = positions.shape[0]
        if layer.cmp is None:
            return torch.empty((batch, 0), dtype=torch.long, device=device), torch.zeros(
                (batch, 0), dtype=torch.bool, device=device
            )
        n_comp = layer.n_comp[slots]
        max_comp = layer.max_comp
        if idx_q is not None:
            k = min(int(self.index_topk), max_comp)
            columns = torch.arange(max_comp, device=device)
            keys = layer.idx[slots[:, None], columns[None, :]]
            dots = torch.bmm(idx_q.float(), keys.float().transpose(1, 2))
            scores = (torch.relu(dots) * idx_w.float().unsqueeze(-1)).sum(dim=1)
            scores = scores * (idx_q.shape[1] * idx_q.shape[2]) ** -0.5
            scores = scores.masked_fill(columns[None, :] >= n_comp[:, None], float("-inf"))
            selected = scores.topk(k, dim=-1).indices
            valid = selected < n_comp[:, None]
            return torch.where(valid, selected, -1).long(), valid
        # HCA: every compressed entry up to the current position is visible.
        k = max_comp
        selected = torch.arange(k, device=device)[None, :].expand(batch, k)
        valid = selected < n_comp[:, None]
        return torch.where(valid, selected, -1).long(), valid

    @torch.no_grad()
    def decode(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """One-token decode step for a batch of slots (in-place cache update)."""
        del input_ids
        tokens = x.shape[0]
        device = x.device
        layer = cache.layers[self.layer_id]
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(tokens, self.n_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.nope_dim, self.rope_head_dim], dim=-1)
        kv = self.kv_norm(self.wkv(x))
        kv_nope, kv_rope = torch.split(kv, [self.nope_dim, self.rope_head_dim], dim=-1)
        q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(1), positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)
        cache.append_kv(self.layer_id, slots, positions, kv)

        idx_q = None
        idx_w = None
        compressor = getattr(self, "compressor", None) or getattr(
            self, "compressor_128", None
        )
        if compressor is not None:
            state_kv = layer.comp_state_kv[slots]
            state_score = layer.comp_state_score[slots]
            compressed, completed = compressor.decode_step(
                x, positions, state_kv, state_score
            )
            layer.comp_state_kv[slots] = state_kv
            layer.comp_state_score[slots] = state_score
            cache.write_comp(self.layer_id, slots, compressed, completed)
            if self.indexer is not None:
                idx_state_kv = layer.idx_state_kv[slots]
                idx_state_score = layer.idx_state_score[slots]
                idx_compressed, _ = self.indexer.compress_decode(
                    x, positions, idx_state_kv, idx_state_score
                )
                layer.idx_state_kv[slots] = idx_state_kv
                layer.idx_state_score[slots] = idx_state_score
                cache.write_idx(self.layer_id, slots, idx_compressed, completed)
                idx_q, idx_w = self.indexer.query(x, qr, positions)

        sparse, sparse_valid = self._select_sparse(
            layer, slots, positions, idx_q, idx_w
        )
        local, local_valid = cache.gather_local(self.layer_id, slots, positions)
        sparse = sparse.to(device)
        if layer.cmp is None:
            sparse_keys = local.new_zeros(local.shape[0], 0, local.shape[-1])
        else:
            sparse_keys = layer.cmp[slots[:, None], sparse.clamp_min(0)]
        keys = torch.cat([local, sparse_keys], dim=1)
        valid = torch.cat([local_valid, sparse_valid], dim=1)
        logits = torch.bmm(q.float(), keys.float().transpose(1, 2))
        logits = logits * self.softmax_scale
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        sink = self.attn_sink.weight.squeeze(-1).float()
        sink_logits = sink.view(1, -1, 1).expand(tokens, -1, 1)
        all_logits = torch.cat([logits, sink_logits], dim=-1)
        probs = torch.softmax(all_logits, dim=-1)[..., :-1]
        out = torch.bmm(probs, keys.float()).to(q.dtype)
        o_nope, o_rope = torch.split(out, [self.nope_dim, self.rope_head_dim], dim=-1)
        o_rope = self.rope(o_rope, positions=positions, inverse=True)
        out = torch.cat([o_nope, o_rope], dim=-1)
        return self.wo_b(self.wo_a(out))
