# Copyright (c) 2026 miniDeepSeek contributors.
"""Slot-based KV cache for incremental DeepSeek-V4 decoding.

The training forward recomputes the whole packed sequence.  Rollout cannot:
GRPO/OPD sample thousands of tokens per step, so decoding needs a cache that
stores

* the raw per-token K/V of every layer (a ring buffer of ``window_size`` slots
  is enough: attention only ever looks at the local window),
* the compressed K/V produced by the CSA/HCA compressor, plus the partial
  block state so a block can be finished on the decode step that completes it,
* the compressed indexer keys for CSA layers (the Lightning Indexer selects
  top-k compressed slots), plus its partial block state.

The cache is *slot addressed*: one row per live sequence.  This is the shape a
data-parallel RL trainer wants (each rank owns a few prompts, group samples
fork the prompt cache), and it lets a batch of sequences with different
lengths decode together.  It is deliberately not paged: at 145M a full slot
is a few MiB, and contiguous rows make the gather logic auditable.

Layouts per layer (``S`` slots, ``W`` window, ``R`` compress ratio):

``kv``            ``[S, W, head_dim]``            ring buffer, index ``t % W``
``cmp``           ``[S, max_comp, head_dim]``     compressed attention K/V
``idx``           ``[S, max_comp, index_dim]``    compressed indexer K (CSA)
``comp_state_kv`` ``[S, state_len, proj_dim]``    partial compressor block
``idx_state_kv``  ``[S, state_len, index_dim*2]`` partial indexer compressor
``length``        ``[S]``                         tokens consumed
``n_comp``        ``[S]``                         compressed entries written

``state_len`` is ``2*R`` for the overlapping compressor (``R == 4``) and ``R``
otherwise; ``proj_dim`` is ``2*head_dim`` for the overlapping case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .config import ModelConfig


@dataclass
class LayerCache:
    """Per-layer cache buffers."""

    layer_id: int
    head_dim: int
    window: int
    compress_ratio: int
    index_dim: int
    kv: Tensor
    cmp: Tensor | None = None
    idx: Tensor | None = None
    comp_state_kv: Tensor | None = None
    comp_state_score: Tensor | None = None
    idx_state_kv: Tensor | None = None
    idx_state_score: Tensor | None = None
    has_compressor: bool = False
    has_indexer: bool = False
    #: Number of compressed entries written for this layer.  Layers with
    #: different ``compress_ratio`` (4 vs 32) have different counts, so this
    #: cannot be a single cache-wide tensor.
    n_comp: Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))

    @property
    def max_comp(self) -> int:
        return 0 if self.cmp is None else int(self.cmp.shape[1])

    @property
    def state_len(self) -> int:
        return 0 if self.comp_state_kv is None else int(self.comp_state_kv.shape[1])

    @property
    def proj_dim(self) -> int:
        return 0 if self.comp_state_kv is None else int(self.comp_state_kv.shape[2])


def _compressor_proj_dim(head_dim: int, ratio: int) -> int:
    # ``Compressor`` uses coff = 2 only for the overlapping ratio-4 window.
    return 2 * head_dim if ratio == 4 else head_dim


def _compressor_state_len(ratio: int) -> int:
    return 2 * ratio if ratio == 4 else ratio


@dataclass
class RolloutCache:
    """A batch of slot-addressed incremental caches.

    All buffers are allocated up front; a finished slot is simply reset
    (``reset``) and reused, so a long rollout step does not allocate.
    """

    num_slots: int
    max_context: int
    device: torch.device
    dtype: torch.dtype
    layers: dict[int, LayerCache] = field(default_factory=dict)
    length: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.length = torch.zeros(self.num_slots, dtype=torch.long, device=self.device)

    @classmethod
    def from_model(
        cls,
        model,
        *,
        num_slots: int,
        max_context: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RolloutCache:
        config: ModelConfig = model.config
        resolved_device = torch.device(device or model.device)
        resolved_dtype = dtype or next(model.parameters()).dtype
        context = int(max_context or model.max_context_length)
        cache = cls(
            num_slots=int(num_slots),
            max_context=context,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        for layer_id, block in enumerate(config.layers):
            attention = block.attention
            ratio = int(attention.compress_ratio)
            has_compressor = ratio > 1
            layer = LayerCache(
                layer_id=layer_id,
                head_dim=int(attention.head_dim),
                window=int(attention.window_size),
                compress_ratio=ratio,
                index_dim=int(attention.index_head_dim),
                kv=torch.zeros(
                    num_slots,
                    int(attention.window_size),
                    int(attention.head_dim),
                    dtype=resolved_dtype,
                    device=resolved_device,
                ),
                has_compressor=has_compressor,
                has_indexer=ratio == 4,
                n_comp=torch.zeros(
                    num_slots, dtype=torch.long, device=resolved_device
                ),
            )
            if has_compressor:
                max_comp = max(context // ratio, 1)
                proj_dim = _compressor_proj_dim(layer.head_dim, ratio)
                state_len = _compressor_state_len(ratio)
                layer.cmp = torch.zeros(
                    num_slots,
                    max_comp,
                    layer.head_dim,
                    dtype=resolved_dtype,
                    device=resolved_device,
                )
                layer.comp_state_kv = torch.zeros(
                    num_slots,
                    state_len,
                    proj_dim,
                    dtype=torch.float32,
                    device=resolved_device,
                )
                layer.comp_state_score = torch.full(
                    (num_slots, state_len, proj_dim),
                    float("-inf"),
                    dtype=torch.float32,
                    device=resolved_device,
                )
                if layer.has_indexer:
                    layer.idx = torch.zeros(
                        num_slots,
                        max_comp,
                        layer.index_dim,
                        dtype=resolved_dtype,
                        device=resolved_device,
                    )
                    idx_proj_dim = _compressor_proj_dim(layer.index_dim, ratio)
                    layer.idx_state_kv = torch.zeros(
                        num_slots,
                        state_len,
                        idx_proj_dim,
                        dtype=torch.float32,
                        device=resolved_device,
                    )
                    layer.idx_state_score = torch.full(
                        (num_slots, state_len, idx_proj_dim),
                        float("-inf"),
                        dtype=torch.float32,
                        device=resolved_device,
                    )
            cache.layers[layer_id] = layer
        return cache

    # -- slot management ------------------------------------------------------
    def reset(self, slots: Tensor) -> None:
        """Clear logical lengths; buffer contents beyond them are never read."""
        self.length[slots] = 0
        for layer in self.layers.values():
            layer.n_comp[slots] = 0

    def copy_slot(self, source: int, target: int) -> None:
        """Fork one slot into another (a prompt cache shared by group samples)."""
        src = int(source)
        dst = int(target)
        self.length[dst] = self.length[src]
        for layer in self.layers.values():
            layer.n_comp[dst] = layer.n_comp[src]
            n = int(layer.n_comp[src].item())
            layer.kv[dst].copy_(layer.kv[src])
            if layer.cmp is not None and n > 0:
                layer.cmp[dst, :n].copy_(layer.cmp[src, :n])
            if layer.idx is not None and n > 0:
                layer.idx[dst, :n].copy_(layer.idx[src, :n])
            if layer.comp_state_kv is not None:
                layer.comp_state_kv[dst].copy_(layer.comp_state_kv[src])
                layer.comp_state_score[dst].copy_(layer.comp_state_score[src])
            if layer.idx_state_kv is not None:
                layer.idx_state_kv[dst].copy_(layer.idx_state_kv[src])
                layer.idx_state_score[dst].copy_(layer.idx_state_score[src])

    # -- prefill --------------------------------------------------------------
    @torch.no_grad()
    def store_prefill(
        self,
        layer_id: int,
        slot: int,
        *,
        kv: Tensor,
        cmp_kv: Tensor | None,
        idx_kv: Tensor | None,
        comp_state: tuple[Tensor, Tensor] | None,
        idx_state: tuple[Tensor, Tensor] | None,
        num_tokens: int,
    ) -> None:
        layer = self.layers[layer_id]
        window = layer.window
        tokens = int(num_tokens)
        if tokens <= window:
            layer.kv[slot, :tokens].copy_(kv)
        else:
            # Keep the last ``window`` tokens in ring order.
            positions = torch.arange(tokens - window, tokens, device=kv.device) % window
            layer.kv[slot, positions] = kv[tokens - window :]
        if cmp_kv is not None and cmp_kv.numel() > 0:
            n = int(cmp_kv.shape[0])
            layer.cmp[slot, :n].copy_(cmp_kv)
            layer.n_comp[slot] += n
        if idx_kv is not None and idx_kv.numel() > 0:
            n = int(idx_kv.shape[0])
            layer.idx[slot, :n].copy_(idx_kv)
        if comp_state is not None:
            layer.comp_state_kv[slot].copy_(comp_state[0])
            layer.comp_state_score[slot].copy_(comp_state[1])
        if idx_state is not None:
            layer.idx_state_kv[slot].copy_(idx_state[0])
            layer.idx_state_score[slot].copy_(idx_state[1])
        if layer_id == 0:
            self.length[slot] = tokens

    # -- decode ---------------------------------------------------------------
    def append_kv(self, layer_id: int, slots: Tensor, positions: Tensor, kv: Tensor) -> None:
        layer = self.layers[layer_id]
        index = positions % layer.window
        layer.kv[slots, index] = kv

    def write_comp(self, layer_id: int, slots: Tensor, values: Tensor) -> None:
        layer = self.layers[layer_id]
        index = layer.n_comp[slots]
        layer.cmp[slots, index] = values.to(layer.cmp.dtype)
        layer.n_comp[slots] = index + 1

    def write_idx(self, layer_id: int, slots: Tensor, values: Tensor) -> None:
        layer = self.layers[layer_id]
        # ``n_comp`` was advanced by ``write_comp`` for the same completed block.
        index = layer.n_comp[slots] - 1
        layer.idx[slots, index] = values.to(layer.idx.dtype)

    def advance(self, slots: Tensor) -> None:
        self.length[slots] += 1

    def gather_local(self, layer_id: int, slots: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``[B, W, D]`` local-window keys and a ``[B, W]`` validity mask."""
        layer = self.layers[layer_id]
        window = layer.window
        offsets = torch.arange(window, device=positions.device)
        index = positions[:, None] - window + 1 + offsets[None, :]
        valid = index >= 0
        ring = index.clamp_min(0) % window
        keys = layer.kv[slots[:, None], ring]
        return keys, valid
