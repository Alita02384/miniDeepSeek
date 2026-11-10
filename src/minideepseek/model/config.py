# Copyright (c) 2026 miniDeepSeek contributors.
# Architecture follows DeepSeek-V4 (torchtitan reference implementation, BSD-3-Clause).
"""Model configuration and architecture-file loading.

The architecture numbers are *data*, not code: every bundled model lives in
``configs/model/<name>.yaml`` as a :class:`~minideepseek.config.ModelSettings`
file.  This module validates that specification, converts it into the runtime
:class:`ModelConfig` tree, and keeps the small amount of derived logic
(hybrid-attention layer map, MTP block, sparse-index top-k) in one place.

Adding a model therefore means adding a YAML file; no Python branch is needed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from ..config import ModelSettings, load_config
from ..paths import CONFIG_DIR

#: Where bundled architecture files live.
MODEL_CONFIG_DIR = CONFIG_DIR / "model"

#: The default model selected by the shipped pretraining/SFT/RL configs.
DEFAULT_MODEL_FLAVOR = "mini_220m_a80m"

#: Bundled flavor names.  ``available_flavors()`` additionally discovers any
#: new YAML dropped into ``configs/model/``.
MODEL_FLAVORS = (
    "debugmodel",
    "mini_220m_a80m",
    "mini_60m_dense_d768",
    "mini_145m_a60m",
)


@dataclass
class AttentionConfig:
    dim: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    n_groups: int
    window_size: int
    compress_ratio: int = 0  # 0 = SWA, 4 = CSA, >1 = HCA
    index_n_heads: int = 8
    index_head_dim: int = 64
    index_topk: int = 64
    hca_topk_max: int | None = None
    #: Longest document this layer was built for.  HCA derives its kernel top-k
    #: from it instead of from the batch, so the TOPK constexpr (and therefore
    #: the compiled kernel) does not change when a pack happens to be shorter.
    max_doc_tokens: int = 4096
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    compress_rope_theta: float = 40000.0
    rope_yarn: bool = False
    compress_rope_yarn: bool = True
    rope_factor: float = 4.0
    original_seq_len: int = 65536
    layer_id: int = 0
    n_layers: int = 1
    use_indexer_aux: bool = False
    indexer_loss_coeff: float = 0.01

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def per_group_in(self) -> int:
        return self.n_heads * self.head_dim // self.n_groups

    @property
    def per_group_out(self) -> int:
        return self.n_groups * self.o_lora_rank

    @property
    def compressor_coff(self) -> int:
        return 2 if self.compress_ratio == 4 else 1

    @property
    def has_indexer(self) -> bool:
        return self.compress_ratio == 4

    @property
    def has_compressor(self) -> bool:
        return self.compress_ratio > 1


@dataclass
class MoEConfig:
    dim: int
    hidden_dim: int
    num_experts: int
    num_shared_experts: int = 1
    top_k: int = 2
    n_hash_layers: int = 3
    layer_id: int = 0
    vocab_size: int = 6400
    route_norm: bool = True
    route_scale: float = 1.5
    load_balance_coeff: float | None = 1e-3
    swiglu_limit: float | None = 10.0


@dataclass
class BlockConfig:
    attention: AttentionConfig
    moe: MoEConfig
    hc_mult: int = 4
    sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    norm_eps: float = 1e-6


@dataclass
class MTPConfig(BlockConfig):
    pass


@dataclass
class ModelConfig:
    dim: int
    vocab_size: int
    max_context_length: int
    layers: list[BlockConfig]
    n_mtp_layers: int = 0
    mtp_layers: list[MTPConfig] = field(default_factory=list)
    hc_mult: int = 4
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    init_std: float = 0.02
    flavor: str = ""

    @property
    def n_layers(self) -> int:
        return len(self.layers)


def available_flavors() -> tuple[str, ...]:
    """Flavor names discoverable from ``configs/model/`` (fallback: bundled)."""
    if MODEL_CONFIG_DIR.is_dir():
        names = tuple(sorted(path.stem for path in MODEL_CONFIG_DIR.glob("*.yaml")))
        if names:
            return names
    return MODEL_FLAVORS


def load_model_settings(flavor: str) -> ModelSettings:
    """Load one bundled architecture file."""
    path = MODEL_CONFIG_DIR / f"{flavor}.yaml"
    if not path.exists():
        raise ValueError(
            f"unknown model flavor {flavor!r}; available={available_flavors()}"
        )
    return load_config(ModelSettings, path)


def build_model_settings(flavor: str, **overrides: Any) -> ModelSettings:
    """Load a bundled architecture and apply run-time overrides."""
    settings = load_model_settings(flavor)
    return dataclasses.replace(settings, **overrides)


def validate_model_settings(settings: ModelSettings) -> None:
    """Reject malformed architecture files before a large model is allocated."""
    if settings.dim <= 0:
        raise ValueError("dim must be positive")
    if settings.n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if settings.n_mtp_layers < 0:
        raise ValueError("n_mtp_layers must be non-negative")
    if settings.n_heads <= 0 or settings.head_dim <= 0:
        raise ValueError("n_heads and head_dim must be positive")
    if settings.head_dim <= settings.rope_head_dim:
        raise ValueError("head_dim must be larger than rope_head_dim")
    if settings.n_groups <= 0 or settings.n_heads % settings.n_groups != 0:
        raise ValueError("n_groups must divide n_heads")
    if len(settings.compress_ratios) != settings.n_layers:
        raise ValueError(
            "compress_ratios must have exactly n_layers entries "
            f"({len(settings.compress_ratios)} != {settings.n_layers})"
        )
    if any(ratio < 0 or ratio == 1 for ratio in settings.compress_ratios):
        raise ValueError("compress_ratios entries must be 0 (SWA) or >= 2")
    if settings.window_size <= 0:
        raise ValueError("window_size must be positive")
    if settings.num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if not 1 <= settings.top_k <= settings.num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    if not 0 <= settings.n_hash_layers <= settings.n_layers:
        raise ValueError("n_hash_layers must be in [0, n_layers]")
    if settings.moe_inter_dim <= 0:
        raise ValueError("moe_inter_dim must be positive")
    if settings.num_shared_experts < 0:
        raise ValueError("num_shared_experts must be non-negative")
    if settings.hc_mult <= 0 or settings.sinkhorn_iters <= 0:
        raise ValueError("hc_mult and sinkhorn_iters must be positive")
    if settings.index_n_heads <= 0 or settings.index_head_dim <= 0:
        raise ValueError("index head settings must be positive")
    if settings.index_head_dim & (settings.index_head_dim - 1):
        raise ValueError("index_head_dim must be a power of two (Hadamard rotation)")
    if settings.seq_len <= 0 or settings.max_context <= 0:
        raise ValueError("seq_len and max_context must be positive")
    if settings.seq_len > settings.max_context:
        raise ValueError(
            f"seq_len {settings.seq_len} exceeds max_context {settings.max_context}"
        )
    if settings.index_topk is not None and settings.index_topk <= 0:
        raise ValueError("index_topk must be positive or null")
    if settings.indexer_loss_coeff < 0:
        raise ValueError("indexer_loss_coeff must be non-negative")


def _index_topk(settings: ModelSettings) -> int:
    """Sparse-index top-k: explicit value, else 1/4 of the context (capped)."""
    if settings.index_topk is not None:
        return int(settings.index_topk)
    return min(2048, max(settings.seq_len // 4, 64))


def _attention_config(
    settings: ModelSettings,
    *,
    compress_ratio: int,
    layer_id: int,
    n_layers: int,
    index_topk: int,
    hca_topk_max: int | None,
    use_indexer_aux: bool,
    indexer_loss_coeff: float,
) -> AttentionConfig:
    return AttentionConfig(
        dim=settings.dim,
        n_heads=settings.n_heads,
        head_dim=settings.head_dim,
        rope_head_dim=settings.rope_head_dim,
        q_lora_rank=settings.q_lora_rank,
        o_lora_rank=settings.o_lora_rank,
        n_groups=settings.n_groups,
        window_size=settings.window_size,
        compress_ratio=compress_ratio,
        index_n_heads=settings.index_n_heads,
        index_head_dim=settings.index_head_dim,
        index_topk=index_topk,
        hca_topk_max=settings.hca_topk_max,
        max_doc_tokens=int(settings.seq_len),
        norm_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=40000.0,
        # SWA keeps the base RoPE; compressed layers rotate with the
        # compressor's YaRN RoPE, matching the reference implementation.
        rope_yarn=False,
        compress_rope_yarn=True,
        layer_id=layer_id,
        n_layers=n_layers,
        use_indexer_aux=use_indexer_aux,
        indexer_loss_coeff=indexer_loss_coeff,
    )


def _moe_config(
    settings: ModelSettings, *, layer_id: int
) -> MoEConfig:
    return MoEConfig(
        dim=settings.dim,
        hidden_dim=settings.moe_inter_dim,
        num_experts=settings.num_experts,
        num_shared_experts=settings.num_shared_experts,
        top_k=settings.top_k,
        n_hash_layers=settings.n_hash_layers,
        layer_id=layer_id,
        vocab_size=settings.vocab_size,
        route_norm=settings.route_norm,
        route_scale=settings.route_scale,
        load_balance_coeff=settings.load_balance_coeff,
        swiglu_limit=settings.swiglu_limit,
    )


def build_model_config_from_settings(settings: ModelSettings) -> ModelConfig:
    """Convert a validated model specification into the runtime dataclass tree."""
    validate_model_settings(settings)
    topk = _index_topk(settings)
    layers: list[BlockConfig] = []
    for layer_id in range(settings.n_layers):
        ratio = int(settings.compress_ratios[layer_id])
        layers.append(
            BlockConfig(
                attention=_attention_config(
                    settings,
                    compress_ratio=ratio,
                    layer_id=layer_id,
                    n_layers=settings.n_layers,
                    index_topk=topk,
                    hca_topk_max=settings.hca_topk_max,
                    use_indexer_aux=ratio == 4 and settings.indexer_loss_coeff > 0,
                    indexer_loss_coeff=settings.indexer_loss_coeff,
                ),
                moe=_moe_config(settings, layer_id=layer_id),
                hc_mult=settings.hc_mult,
                sinkhorn_iters=settings.sinkhorn_iters,
                hc_eps=1e-6,
                norm_eps=1e-6,
            )
        )

    mtp_layers: list[MTPConfig] = []
    for depth in range(settings.n_mtp_layers):
        layer_id = settings.n_layers + depth
        # The MTP head is a single SWA transformer block, like the reference
        # V4 MTP layer.  It has no compressor/indexer and therefore no
        # auxiliary distillation loss.
        mtp_layers.append(
            MTPConfig(
                attention=_attention_config(
                    settings,
                    compress_ratio=0,
                    layer_id=layer_id,
                    n_layers=settings.n_layers + settings.n_mtp_layers,
                    index_topk=topk,
                    hca_topk_max=settings.hca_topk_max,
                    use_indexer_aux=False,
                    indexer_loss_coeff=0.0,
                ),
                moe=_moe_config(settings, layer_id=layer_id),
                hc_mult=settings.hc_mult,
                sinkhorn_iters=settings.sinkhorn_iters,
                hc_eps=1e-6,
                norm_eps=1e-6,
            )
        )

    return ModelConfig(
        dim=settings.dim,
        vocab_size=settings.vocab_size,
        max_context_length=settings.max_context,
        layers=layers,
        n_mtp_layers=settings.n_mtp_layers,
        mtp_layers=mtp_layers,
        hc_mult=settings.hc_mult,
        norm_eps=1e-6,
        tie_word_embeddings=settings.tie_word_embeddings,
        flavor=settings.flavor,
    )


class _Unset:
    """Sentinel: the caller did not pass this argument."""


UNSET = _Unset()


def build_model_config(
    flavor: str,
    *,
    seq_len: int | None = None,
    max_context: int | None = None,
    n_mtp_layers: int | None = None,
    tie_word_embeddings: bool | None = None,
    swiglu_limit: Any = UNSET,
    index_topk: int | None = None,
    indexer_loss_coeff: Any = UNSET,
) -> ModelConfig:
    """Build a config for a bundled flavor, with optional run-time overrides.

    ``seq_len`` sets the training length; when it is given without an explicit
    ``max_context`` the context limit follows it (the historical behavior used
    by the smoke tests).  Training runs pass a fully resolved ``ModelSettings``
    to :func:`build_model_config_from_settings` instead, so ``seq_len`` (4K)
    and ``max_context`` (8K) can differ.
    """
    settings = load_model_settings(flavor)
    updates: dict[str, Any] = {}
    if seq_len is not None:
        updates["seq_len"] = int(seq_len)
    if max_context is not None:
        updates["max_context"] = int(max_context)
    elif seq_len is not None:
        updates["max_context"] = int(seq_len)
    if n_mtp_layers is not None:
        updates["n_mtp_layers"] = int(n_mtp_layers)
    if tie_word_embeddings is not None:
        updates["tie_word_embeddings"] = bool(tie_word_embeddings)
    if swiglu_limit is not UNSET:
        updates["swiglu_limit"] = swiglu_limit
    if index_topk is not None:
        updates["index_topk"] = int(index_topk)
    if indexer_loss_coeff is not UNSET:
        updates["indexer_loss_coeff"] = float(indexer_loss_coeff)
    if updates:
        settings = dataclasses.replace(settings, **updates)
    return build_model_config_from_settings(settings)
