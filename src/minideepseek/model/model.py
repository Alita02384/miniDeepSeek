# Copyright (c) 2026 miniDeepSeek contributors.
# Decoder/MTP structure refactored from torchtitan/models/deepseek_v4/model.py
# and deepseek_v3/mtp.py (BSD-3-Clause, Meta).
"""Self-contained DeepSeek-V4 decoder with mHC and MTP."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as _checkpoint

from .attention import DeepSeekV4Attention, PackedAttentionMetadata
from .config import BlockConfig, ModelConfig, MTPConfig
from .mhc import HcHead, HcPost, HcPre
from .moe import MoE
from .nn import RMSNorm, init_output_linear


def _cu_seqlens_from_positions(positions: torch.Tensor) -> torch.Tensor:
    starts = (positions == 0).nonzero(as_tuple=False).flatten().to(torch.int32)
    if starts.numel() == 0 or int(starts[0]) != 0:
        raise ValueError("positions must start a document at index 0")
    return torch.cat([starts, torch.tensor([positions.numel()], dtype=torch.int32, device=positions.device)])


def roll_mtp_sequence(
    sequence: torch.Tensor,
    *,
    shift: int,
    fill_value: int | bool,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-roll a sequence without crossing document boundaries."""
    total = sequence.shape[0]
    if shift <= 0 or shift > total:
        raise ValueError(f"shift must be in [1, {total}]")
    shifted = torch.full_like(sequence, fill_value)
    valid = torch.zeros_like(sequence, dtype=torch.bool)
    source = sequence[shift:]
    valid_tokens = torch.ones_like(source, dtype=torch.bool)
    if positions is not None:
        valid_tokens &= positions[shift:] == positions[: total - shift] + shift
    shifted[: total - shift] = torch.where(valid_tokens, source, shifted[: total - shift])
    valid[: total - shift] = valid_tokens
    return shifted, valid


class DeepSeekV4TransformerBlock(nn.Module):
    def __init__(self, config: BlockConfig, *, max_context_length: int):
        super().__init__()
        self.attention = DeepSeekV4Attention(config.attention, max_context_length=max_context_length)
        self.attention_norm = RMSNorm(config.attention.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.attention.dim, config.norm_eps)
        self.feed_forward = MoE(config.moe)
        self.hc_attn_pre = HcPre(
            config.attention.dim, config.hc_mult, config.sinkhorn_iters, config.hc_eps, config.norm_eps
        )
        self.hc_ffn_pre = HcPre(
            config.attention.dim, config.hc_mult, config.sinkhorn_iters, config.hc_eps, config.norm_eps
        )
        self.hc_post = HcPost()

    def init_weights(self) -> None:
        self.attention.init_weights()
        nn.init.ones_(self.attention_norm.weight)
        nn.init.ones_(self.ffn_norm.weight)
        self.feed_forward.init_weights()
        self.hc_attn_pre.init_weights()
        self.hc_ffn_pre.init_weights()

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        metadata: PackedAttentionMetadata,
        positions: torch.Tensor,
        *,
        cache=None,
        slot: int | None = None,
    ) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_attn_pre(x)
        x = self.attention(
            self.attention_norm(x),
            input_ids,
            positions,
            metadata,
            cache=cache,
            slot=slot,
        )
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_ffn_pre(x)
        if getattr(self.feed_forward, "router", None) is not None and self.feed_forward.router.hash:
            x = self.feed_forward(self.ffn_norm(x), input_ids=input_ids)
        else:
            x = self.feed_forward(self.ffn_norm(x))
        x = self.hc_post(x, residual, post, comb)
        return x

    def decode(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """One-token incremental step (no metadata / no MTP)."""
        residual = x
        x, post, comb = self.hc_attn_pre(x)
        x = self.attention.decode(
            self.attention_norm(x), input_ids, positions, cache, slots
        )
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_ffn_pre(x)
        if getattr(self.feed_forward, "router", None) is not None and self.feed_forward.router.hash:
            x = self.feed_forward(self.ffn_norm(x), input_ids=input_ids)
        else:
            x = self.feed_forward(self.ffn_norm(x))
        x = self.hc_post(x, residual, post, comb)
        return x


class MTPBlock(DeepSeekV4TransformerBlock):
    def __init__(self, config: MTPConfig, *, max_context_length: int):
        super().__init__(config, max_context_length=max_context_length)
        dim = config.attention.dim
        self.enorm = RMSNorm(dim, config.norm_eps)
        self.hnorm = RMSNorm(dim, config.norm_eps)
        self.e_proj = nn.Linear(dim, dim, bias=False)
        self.h_proj = nn.Linear(dim, dim, bias=False)
        self.mtp_norm = RMSNorm(dim, config.norm_eps)
        self.hc_head = HcHead(dim, config.hc_mult, config.norm_eps, config.hc_eps)

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.ones_(self.enorm.weight)
        nn.init.ones_(self.hnorm.weight)
        nn.init.ones_(self.mtp_norm.weight)
        for module in (self.e_proj, self.h_proj):
            nn.init.trunc_normal_(module.weight, std=0.02)
        self.hc_head.init_weights()

    def forward(
        self,
        mtp_input_embed: torch.Tensor,
        prev_hc_hidden: torch.Tensor,
        mtp_input_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        metadata: PackedAttentionMetadata,
        positions: torch.Tensor,
    ):
        prev = prev_hc_hidden * valid_mask.view(-1, 1, 1).to(prev_hc_hidden.dtype)
        hidden = self.e_proj(self.enorm(mtp_input_embed)).unsqueeze(1)
        hidden = hidden + self.h_proj(self.hnorm(prev))
        next_hc = super().forward(hidden, mtp_input_ids, metadata, positions)
        prediction = self.mtp_norm(self.hc_head(next_hc))
        return next_hc, prediction


class DeepSeekV4Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.vocab_size = config.vocab_size
        self.hc_mult = config.hc_mult
        self.n_mtp_layers = config.n_mtp_layers
        self.max_context_length = config.max_context_length
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.hc_head = HcHead(config.dim, config.hc_mult, config.norm_eps)
        self.layers = nn.ModuleDict(
            {
                str(index): DeepSeekV4TransformerBlock(layer, max_context_length=config.max_context_length)
                for index, layer in enumerate(config.layers)
            }
        )
        self.mtp_layers = nn.ModuleList(
            [
                MTPBlock(layer, max_context_length=config.max_context_length)
                for layer in config.mtp_layers
            ]
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight
        # Activation checkpointing (per block).  Off by default: it trades ~30%
        # extra compute for a large drop in peak activation memory, which is
        # what makes long-context stages fit on 24 GB cards.  The flag is set by
        # the training config (``train.activation_checkpointing``).
        self.grad_checkpointing = False

    def _run_block(self, block: nn.Module, *args, **kwargs):
        """Call one transformer block, recomputing it in backward if enabled.

        Non-reentrant checkpointing keeps ``no_grad``/autocast behaviour intact
        and works with FSDP2 (the recompute re-issues the all-gather).
        """
        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            return _checkpoint(block, *args, use_reentrant=False, **kwargs)
        return block(*args, **kwargs)


    def collect_indexer_loss(self) -> torch.Tensor | None:
        """Sum the CSA indexer distillation losses from the last forward."""
        losses = [
            layer.attention.last_indexer_loss
            for layer in self.layers.values()
            if getattr(layer.attention, "last_indexer_loss", None) is not None
        ]
        if not losses:
            return None
        total = losses[0]
        for loss in losses[1:]:
            total = total + loss
        return total

    def init_weights(self) -> None:
        if self.config.tie_word_embeddings:
            # The reference V4 keeps an untied std=0.02 lm_head next to a
            # std=1.0 embedding.  Tying shares one matrix, so the output role
            # fixes the logit scale; use the output-projection init to keep
            # the initial cross-entropy near log(vocab) instead of ~750.
            std = self.dim**-0.5
            nn.init.trunc_normal_(
                self.tok_embeddings.weight, std=std, a=-3 * std, b=3 * std
            )
        else:
            nn.init.normal_(self.tok_embeddings.weight, std=1.0)
            init_output_linear(self.lm_head, self.dim)
        nn.init.ones_(self.norm.weight)
        self.hc_head.init_weights()
        for layer in self.layers.values():
            layer.init_weights()
        for layer in self.mtp_layers:
            layer.init_weights()

    def _get_attention_metadata(
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ) -> PackedAttentionMetadata:
        del padding_mask, max_num_documents
        cu_seqlens = _cu_seqlens_from_positions(positions)
        lengths = torch.diff(cu_seqlens)
        max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
        if max_context_length is not None and max_seqlen > max_context_length:
            raise ValueError(f"sequence length {max_seqlen} exceeds {max_context_length}")
        return PackedAttentionMetadata(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    def mtp_forward(
        self,
        prev_hc_hidden: torch.Tensor,
        tokens: torch.Tensor,
        metadata: PackedAttentionMetadata,
        positions: torch.Tensor,
    ):
        outputs = []
        for depth, block in enumerate(self.mtp_layers, 1):
            mtp_tokens, valid = roll_mtp_sequence(
                tokens, shift=depth, fill_value=0, positions=positions
            )
            prev_hc_hidden, prediction = self._run_block(
                block,
                self.tok_embeddings(mtp_tokens),
                prev_hc_hidden,
                mtp_tokens.detach().long(),
                valid,
                metadata,
                positions,
            )
            outputs.append(prediction)
        return outputs

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: PackedAttentionMetadata | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
        *,
        cache=None,
        slot: int | None = None,
        use_mtp: bool = True,
    ):
        del padding_mask, aux_loss_denominators
        if positions is None:
            positions = torch.arange(tokens.numel(), device=tokens.device)
        if attention_metadata is None:
            attention_metadata = self._get_attention_metadata(positions)
        h = self.tok_embeddings(tokens)
        h = h.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for layer in self.layers.values():
            h = self._run_block(
                layer,
                h,
                tokens,
                attention_metadata,
                positions,
                cache=cache,
                slot=slot,
            )
        prev_hc = h
        hidden = self.hc_head(h)
        hidden = self.norm(hidden)
        main = self.lm_head(hidden) if self.lm_head is not None else hidden
        if not self.mtp_layers or not use_mtp:
            return main
        outputs = [hidden] + self.mtp_forward(prev_hc, tokens, attention_metadata, positions)
        return tuple(self.lm_head(item) for item in outputs)

    @torch.no_grad()
    def decode(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cache,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """One token per slot: return ``[B, vocab]`` main-depth logits."""
        h = self.tok_embeddings(tokens)
        h = h.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for layer in self.layers.values():
            h = layer.decode(h, tokens, positions, cache, slots)
        hidden = self.norm(self.hc_head(h))
        cache.advance(slots)
        return self.lm_head(hidden)
