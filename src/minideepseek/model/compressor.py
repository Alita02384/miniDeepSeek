# Copyright (c) 2026 miniDeepSeek contributors.
# Refactored from torchtitan/models/deepseek_v4/compressor.py (BSD-3-Clause, Meta).
"""KV compression and the CSA indexer.

Implements V4 report §2.3.1 (compressed KV state, overlap/non-overlap blocks)
and the lightning indexer trained through the KL distillation loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .nn import RMSNorm, hadamard, init_linear
from .rope import RoPE


def compressed_cu_seqlens(cu_seqlens: torch.Tensor, ratio: int) -> torch.Tensor:
    """Offsets of each document's compressed entries (``len // ratio`` each)."""
    counts = torch.diff(cu_seqlens) // ratio
    return F.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))


def compressed_gather_indices(
    topk_indices: torch.Tensor, cu_seqlens: torch.Tensor | None, ratio: int
) -> torch.Tensor:
    gather = topk_indices.clamp_min(0).long()
    if cu_seqlens is not None:
        doc = torch.searchsorted(
            cu_seqlens[1:],
            torch.arange(topk_indices.shape[0], device=topk_indices.device),
            right=True,
            out_int32=True,
        )
        gather = gather + compressed_cu_seqlens(cu_seqlens, ratio)[doc, None]
    return gather.masked_fill(topk_indices < 0, 0)


class Compressor(nn.Module):
    """Turn each ``ratio`` tokens into one weighted-sum KV entry."""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        *,
        overlap: bool | None = None,
        rope: RoPE,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4 if overlap is None else overlap
        coff = 2 if self.overlap else 1
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
        self.norm = RMSNorm(head_dim, norm_eps)
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim))
        self.rope = rope

    def init_weights(self) -> None:
        init_linear(self.wkv)
        init_linear(self.wgate)
        nn.init.ones_(self.norm.weight)
        nn.init.trunc_normal_(self.ape, std=0.02)

    def _overlap_transform(self, tensor: torch.Tensor, value: float, first: torch.Tensor | None):
        d = self.head_dim
        prev = torch.cat([torch.full_like(tensor[:1, :, :d], value), tensor[:-1, :, :d]], dim=0)
        if first is not None:
            prev = prev.masked_fill(first[:, None, None], value)
        return torch.cat([prev, tensor[:, :, d:]], dim=1)

    # -- incremental decoding helpers -------------------------------------
    # The training forward compresses a whole sequence at once.  Rollout needs
    # the same arithmetic one token (or one prefill chunk) at a time: store the
    # per-token projections of the unfinished block, and pool when it closes.
    # The state layout follows the official inference implementation
    # (DeepSeek-V4-Flash/inference/model.py: Compressor.kv_state/score_state).
    def project_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token ``wkv``/``wgate`` projections in fp32 (no pooling)."""
        with torch.autocast(
            device_type="cuda", dtype=torch.float32, enabled=x.device.type == "cuda"
        ):
            kv = self.wkv(x)
            score = self.wgate(x)
        return kv.float(), score.float()

    def _finalize(self, pooled: torch.Tensor, block_positions: torch.Tensor, dtype) -> torch.Tensor:
        """Norm + RoPE the pooled block K/V, matching ``forward``."""
        kv = self.norm(pooled.to(dtype))
        rd = self.rope_head_dim
        nope, rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        rope = self.rope(rope.unsqueeze(1), positions=block_positions).squeeze(1)
        return torch.cat([nope, rope], dim=-1)

    @torch.no_grad()
    def capture_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """State left after a single-document prefill of ``x``.

        ``forward`` only emits whole blocks; the tail tokens (and, for the
        overlapping ratio-4 compressor, the previous block's first half) are
        kept here so the next decode step can finish the block.
        """
        seqlen = int(x.shape[0])
        ratio = self.compress_ratio
        kv, score = self.project_tokens(x)
        state_len = 2 * ratio if self.overlap else ratio
        proj_dim = kv.shape[-1]
        state_kv = torch.zeros(state_len, proj_dim, dtype=torch.float32, device=x.device)
        state_score = torch.full(
            (state_len, proj_dim), float("-inf"), dtype=torch.float32, device=x.device
        )
        cutoff = (seqlen // ratio) * ratio
        remainder = seqlen - cutoff
        if self.overlap:
            if cutoff >= ratio:
                state_kv[:ratio] = kv[cutoff - ratio : cutoff]
                state_score[:ratio] = score[cutoff - ratio : cutoff] + self.ape
            if remainder > 0:
                state_kv[ratio : ratio + remainder] = kv[cutoff:]
                state_score[ratio : ratio + remainder] = score[cutoff:] + self.ape[:remainder]
        elif remainder > 0:
            state_kv[:remainder] = kv[cutoff:]
            state_score[:remainder] = score[cutoff:] + self.ape[:remainder]
        return state_kv, state_score

    @torch.no_grad()
    def decode_step(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        state_kv: torch.Tensor,
        state_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulate one token per slot and pool the block it completes.

        Args:
            x: ``[B, dim]`` current-token attention input.
            positions: ``[B]`` absolute positions (0-based, within the document).
            state_kv/state_score: ``[B, state_len, proj_dim]`` partial blocks,
                updated in place.

        Returns:
            ``(compressed [B, head_dim], completed [B] bool)``.  ``compressed``
            is only meaningful where ``completed`` is true; non-completing rows
            contain garbage and must be masked by the caller.
        """
        ratio = self.compress_ratio
        kv, score = self.project_tokens(x)
        pos = positions.long()
        offset = pos % ratio
        rows = torch.arange(x.shape[0], device=x.device)
        index = ratio + offset if self.overlap else offset
        state_kv[rows, index] = kv
        state_score[rows, index] = score + self.ape[offset]
        completed = (pos + 1) % ratio == 0
        if self.overlap:
            d = self.head_dim
            block_kv = torch.cat(
                [state_kv[:, :ratio, :d], state_kv[:, ratio:, d:]], dim=1
            )
            block_score = torch.cat(
                [state_score[:, :ratio, :d], state_score[:, ratio:, d:]], dim=1
            )
        else:
            block_kv = state_kv[:, :ratio]
            block_score = state_score[:, :ratio]
        pooled = (block_kv * block_score.softmax(dim=1)).sum(dim=1)
        block_positions = pos + 1 - ratio
        compressed = self._finalize(pooled, block_positions, x.dtype)
        if self.overlap:
            # Shift only the rows whose block closed; ``torch.where`` keeps the
            # shape static (a ``nonzero`` index would break CUDA-graph capture).
            shift = completed[:, None, None]
            state_kv[:, :ratio] = torch.where(shift, state_kv[:, ratio:], state_kv[:, :ratio])
            state_score[:, :ratio] = torch.where(
                shift, state_score[:, ratio:], state_score[:, :ratio]
            )
        return compressed, completed

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cu_seqlens: torch.Tensor):
        seqlen = x.shape[0]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        # The reference keeps the compressor projections in fp32: the pooled
        # KV then goes through a bf16/fp16 autocast.  Doing the projections in
        # low precision changes this module's numerics on CUDA.
        with torch.autocast(
            device_type="cuda", dtype=torch.float32, enabled=x.device.type == "cuda"
        ):
            kv = self.wkv(x)
            score = self.wgate(x)
        if cu_seqlens is None:
            if seqlen % ratio != 0:
                raise ValueError(f"seqlen {seqlen} not divisible by ratio {ratio}")
            comp_positions = positions[::ratio]
            kv = kv.unflatten(0, (-1, ratio))
            score = score.unflatten(0, (-1, ratio))
            first = None
        else:
            if seqlen < ratio:
                return x.new_zeros(0, self.head_dim)
            cmp_cu = compressed_cu_seqlens(cu_seqlens, ratio)
            slot = torch.arange(seqlen // ratio, dtype=torch.int32, device=x.device)
            doc = torch.searchsorted(cmp_cu[1:], slot, right=True, out_int32=True)
            group = slot - cmp_cu[doc]
            start = torch.where(slot < cmp_cu[-1], cu_seqlens[doc] + group * ratio, 0)
            token = start.unsqueeze(1) + torch.arange(ratio, device=x.device)
            kv, score = kv[token], score[token]
            comp_positions = positions[start]
            first = group == 0
        score = score + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0.0, first)
            score = self._overlap_transform(score, float("-inf"), first)
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = self.norm(kv.to(x.dtype))
        nope, rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        rope = self.rope(rope.unsqueeze(1), positions=comp_positions).squeeze(1)
        return torch.cat([nope, rope], dim=-1)


class Indexer(nn.Module):
    """Produce low-dimensional query/key features and top-k compressed slots."""

    def __init__(
        self,
        dim: int,
        q_lora_rank: int,
        num_heads: int,
        index_head_dim: int,
        compress_ratio: int,
        rope: RoPE,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope.dim
        self.softmax_scale = index_head_dim**-0.5
        self.compress_ratio = compress_ratio
        self.rope = rope
        self.wq_b = nn.Linear(q_lora_rank, num_heads * index_head_dim, bias=False)
        self.weights_proj = nn.Linear(dim, num_heads, bias=False)
        self.compressor = Compressor(
            dim,
            index_head_dim,
            rope.dim,
            compress_ratio,
            overlap=True,
            rope=rope,
            norm_eps=norm_eps,
        )

    def init_weights(self) -> None:
        init_linear(self.wq_b)
        init_linear(self.weights_proj)
        self.compressor.init_weights()

    def _rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        matrix = hadamard(x.shape[-1], x.dtype, x.device)
        return F.linear(x, matrix) * (x.shape[-1] ** -0.5)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        *,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ):
        q = self.wq_b(qr).view(x.shape[0], self.num_heads, self.head_dim)
        nope, rope = torch.split(q, [self.head_dim - self.rope_head_dim, self.rope_head_dim], dim=-1)
        rope = self.rope(rope, positions=positions)
        q = self._rotate_activation(torch.cat([nope, rope], dim=-1))
        k = self._rotate_activation(
            self.compressor(x, positions=positions, cu_seqlens=cu_seqlens)
        )
        w = self.weights_proj(x) * (self.softmax_scale * self.num_heads**-0.5)
        return q, k, w

    # -- incremental query/key helpers ------------------------------------
    def query(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Indexer query ``[B, heads, head_dim]`` and per-head weight ``[B, heads]``."""
        q = self.wq_b(qr).view(x.shape[0], self.num_heads, self.head_dim)
        nope, rope = torch.split(
            q, [self.head_dim - self.rope_head_dim, self.rope_head_dim], dim=-1
        )
        rope = self.rope(rope, positions=positions)
        q = self._rotate_activation(torch.cat([nope, rope], dim=-1))
        w = self.weights_proj(x) * (self.softmax_scale * self.num_heads**-0.5)
        return q, w

    def compress_decode(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        state_kv: torch.Tensor,
        state_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressed, completed = self.compressor.decode_step(
            x, positions, state_kv, state_score
        )
        return self._rotate_activation(compressed), completed

    @staticmethod
    def select(
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_w: torch.Tensor,
        *,
        topk: int,
        ratio: int,
        cu_seqlens: torch.Tensor,
        return_scores: bool,
        query_chunk: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reference Lightning-Indexer selection (no fused kernel dependency).

        Queries are processed in chunks: the naive ``[T, H, S]`` score tensor
        is 7 GB for a 30k-token RL mini-batch (S is the compressed length), and
        it is the OOM that first showed up when GRPO moved to 16-sample actor
        micro-batches.  Chunking keeps the same top-k per query.
        """
        tokens = idx_q.shape[0]
        candidates = idx_k.shape[0]
        cu_k = compressed_cu_seqlens(cu_seqlens, ratio)
        k = min(topk, max(candidates, 1))
        if candidates == 0 or k == 0:
            empty = torch.full((tokens, topk), -1, dtype=torch.int32, device=idx_q.device)
            return empty, None
        accum = torch.float32
        idx_k_acc = idx_k.to(accum)
        idx_w_acc = idx_w.to(accum)
        scale = (idx_q.shape[1] * idx_q.shape[2]) ** -0.5
        selected_out = torch.empty(tokens, topk, dtype=torch.int32, device=idx_q.device)
        scores_out = (
            torch.empty(tokens, topk, dtype=torch.float32, device=idx_q.device)
            if return_scores
            else None
        )
        columns = torch.arange(candidates, device=idx_q.device)
        chunk = max(int(query_chunk), 1)
        for start in range(0, tokens, chunk):
            stop = min(start + chunk, tokens)
            positions = torch.arange(start, stop, device=idx_q.device, dtype=torch.int32)
            doc = torch.searchsorted(cu_seqlens[1:], positions, right=True, out_int32=True)
            starts = cu_k[doc]
            local = positions - cu_seqlens[doc]
            ends = torch.minimum(
                cu_k[(doc + 1).clamp(max=cu_k.shape[0] - 1)], starts + (local + 1) // ratio
            )
            valid = (columns[None, :] >= starts[:, None]) & (
                columns[None, :] < ends[:, None]
            )
            dots = torch.einsum(
                "qhd,sd->qhs", idx_q[start:stop].to(accum), idx_k_acc
            )
            scores = (torch.relu(dots) * idx_w_acc[start:stop].unsqueeze(-1)).sum(1) * scale
            scores = scores.masked_fill(~valid, float("-inf"))
            selected = scores.topk(k, dim=-1).indices.to(torch.int32)
            selected_valid = (selected >= starts[:, None]) & (selected < ends[:, None])
            local_indices = torch.where(selected_valid, selected - starts[:, None], -1)
            if k < topk:
                local_indices = F.pad(local_indices, (0, topk - k), value=-1)
            selected_out[start:stop] = local_indices
            if return_scores:
                gather = compressed_gather_indices(local_indices, cu_seqlens, ratio)
                selected_k = idx_k[gather]
                logits = torch.einsum("qhd,qkd->qhk", idx_q[start:stop], selected_k)
                topk_scores = (torch.relu(logits) * idx_w[start:stop].unsqueeze(-1)).sum(1)
                topk_scores = topk_scores.masked_fill(local_indices < 0, -torch.inf)
                scores_out[start:stop] = topk_scores
        return selected_out, scores_out


class SparseIndexerLoss(nn.Module):
    """Distill the CSA indexer on the main attention's log-sum-exp.

    The teacher distribution is the head-averaged, L1-normalized attention mass
    that the main kernel put on each selected compressed slot (DeepSeek-V3.2
    section 2.1, inherited by V4).  The student is the indexer's selected-slot
    logits.  The returned scalar is ``coeff / denominator`` times the summed
    KL, matching the reference's carrier injection.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        compress_ratio: int = 4,
        coeff: float = 0.01,
        chunk_size: int = 128,
        mass_weighted: bool = False,
        softmax_scale: float,
    ):
        super().__init__()
        if coeff < 0:
            raise ValueError("indexer loss coefficient must be non-negative")
        self.num_heads = num_heads
        self.compress_ratio = compress_ratio
        self.coeff = coeff
        self.chunk_size = chunk_size
        self.mass_weighted = mass_weighted
        self.softmax_scale = softmax_scale

    @torch.no_grad()
    def teacher(
        self,
        query: torch.Tensor,
        compressed_kv: torch.Tensor,
        topk_indices: torch.Tensor,
        attention_lse: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        gather = compressed_gather_indices(topk_indices, cu_seqlens, self.compress_ratio)
        targets = []
        with torch.autocast(device_type=query.device.type, enabled=False):
            for start in range(0, query.shape[0], self.chunk_size):
                stop = start + self.chunk_size
                valid = topk_indices[start:stop] >= 0
                keys = compressed_kv[gather[start:stop]].float()
                logits = (
                    torch.einsum("thd,tkd->thk", query[start:stop].float(), keys)
                    * self.softmax_scale
                )
                logits = logits - attention_lse[start:stop, :, None].float()
                logits = logits.masked_fill(~valid[:, None], float("-inf"))
                targets.append(logits.exp().sum(dim=1) / self.num_heads)
        return torch.cat(targets)

    def forward(
        self,
        query: torch.Tensor,
        compressed_kv: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        attention_lse: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        denominator: int | torch.Tensor,
    ) -> torch.Tensor:
        if topk_scores.numel() == 0:
            return topk_scores.new_zeros(())
        p = self.teacher(query, compressed_kv, topk_indices, attention_lse, cu_seqlens)
        t = p / p.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        slot_valid = torch.isfinite(topk_scores)
        logits = topk_scores.float()
        all_invalid = ~slot_valid.any(dim=-1, keepdim=True)
        log_student = F.log_softmax(
            logits.masked_fill(~slot_valid, float("-inf")).masked_fill(all_invalid, 0.0),
            dim=-1,
        )
        log_student = log_student.masked_fill(~slot_valid, 0.0)
        weight = p if self.mass_weighted else t
        terms = torch.special.xlogy(weight, t) - weight * log_student
        raw_sum = terms.masked_fill(~slot_valid, 0.0).sum()
        scale = 1.0 / max(float(denominator), 1.0)
        return raw_sum * (self.coeff * scale)
