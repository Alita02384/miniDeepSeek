# Copyright (c) 2026 miniDeepSeek contributors.
# Architecture follows DeepSeek-V4 (torchtitan reference implementation, BSD-3-Clause).
"""Model configuration and the miniDeepSeek flavors.

Everything here is plain Python/PyTorch: the project is self-contained and does
not import the reference checkouts at runtime.  See ``doc/sources.md`` for the
upstream files this design was refactored from.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


def _attention(
    *,
    dim: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    n_groups: int,
    compress_ratio: int,
    window_size: int,
    index_topk: int,
    layer_id: int,
    n_layers: int,
    norm_eps: float,
    rope_yarn: bool = False,
    compress_rope_yarn: bool = True,
    indexer_loss_coeff: float = 0.01,
) -> AttentionConfig:
    return AttentionConfig(
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        n_groups=n_groups,
        window_size=window_size,
        compress_ratio=compress_ratio,
        index_n_heads=8,
        index_head_dim=64,
        index_topk=index_topk,
        norm_eps=norm_eps,
        rope_theta=10000.0,
        compress_rope_theta=40000.0,
        rope_yarn=rope_yarn,
        compress_rope_yarn=compress_rope_yarn,
        layer_id=layer_id,
        n_layers=n_layers,
        use_indexer_aux=indexer_loss_coeff > 0,
        indexer_loss_coeff=indexer_loss_coeff,
    )


def _moe(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    n_hash_layers: int,
    layer_id: int,
    vocab_size: int,
    route_norm: bool,
    route_scale: float,
    load_balance_coeff: float | None,
    swiglu_limit: float | None,
) -> MoEConfig:
    return MoEConfig(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
        n_hash_layers=n_hash_layers,
        layer_id=layer_id,
        vocab_size=vocab_size,
        route_norm=route_norm,
        route_scale=route_scale,
        load_balance_coeff=load_balance_coeff,
        swiglu_limit=swiglu_limit,
    )


_DEBUG_PRESET = dict(
    dim=256,
    n_layers=4,
    n_heads=16,
    head_dim=256,
    moe_inter_dim=256,
    q_lora_rank=128,
    o_lora_rank=128,
    n_groups=2,
    compress_ratios=(0, 0, 4, 128),
    window_size=16,
    index_topk=16,
    n_hash_layers=2,
    top_k=3,
    route_norm=False,
    num_experts=4,
    num_shared_experts=1,
    vocab_size=2048,
    rope_yarn=False,
)

_MINI_PRESETS = {
    "mini_60m": dict(
        dim=640, n_layers=12, n_heads=10, head_dim=64, moe_inter_dim=256,
        q_lora_rank=256, o_lora_rank=256, n_groups=2,
        compress_ratios=(0, 0) + (4, 32) * 5, window_size=128,
        index_topk=2048, n_hash_layers=3, top_k=2, route_norm=True,
        num_experts=6, num_shared_experts=1, vocab_size=6400, rope_yarn=False,
    ),
    "mini_110m": dict(
        dim=768, n_layers=16, n_heads=12, head_dim=64, moe_inter_dim=320,
        q_lora_rank=288, o_lora_rank=288, n_groups=2,
        compress_ratios=(0, 0) + (4, 32) * 7, window_size=128,
        index_topk=2048, n_hash_layers=3, top_k=2, route_norm=True,
        num_experts=6, num_shared_experts=1, vocab_size=6400, rope_yarn=False,
    ),
    "mini_200m": dict(
        dim=1024, n_layers=16, n_heads=16, head_dim=64, moe_inter_dim=416,
        q_lora_rank=384, o_lora_rank=384, n_groups=4,
        compress_ratios=(0, 0) + (4, 32) * 7, window_size=128,
        index_topk=2048, n_hash_layers=3, top_k=2, route_norm=True,
        num_experts=6, num_shared_experts=1, vocab_size=6400, rope_yarn=False,
    ),
}


def _build_preset(
    preset: dict,
    *,
    seq_len: int,
    n_mtp_layers: int,
    dense: bool = False,
    tie_word_embeddings: bool = True,
    swiglu_limit: float | None = 10.0,
    index_topk: int | None = None,
    indexer_loss_coeff: float = 0.01,
) -> ModelConfig:
    dim = preset["dim"]
    n_layers = preset["n_layers"]
    n_heads = preset["n_heads"]
    head_dim = preset["head_dim"]
    compress_ratios = (0,) * n_layers if dense else preset["compress_ratios"]
    if len(compress_ratios) < n_layers:
        raise ValueError("compress_ratios shorter than n_layers")
    window_size = seq_len if dense else preset["window_size"]
    if dense:
        topk = 64
    elif index_topk is not None:
        topk = int(index_topk)
    else:
        topk = min(int(preset["index_topk"]), max(seq_len // 4, 64))
    layers = []
    for layer_id in range(n_layers):
        ratio = compress_ratios[layer_id]
        layers.append(
            BlockConfig(
                attention=_attention(
                    dim=dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    rope_head_dim=32,
                    q_lora_rank=preset["q_lora_rank"],
                    o_lora_rank=preset["o_lora_rank"],
                    n_groups=preset["n_groups"],
                    compress_ratio=ratio,
                    window_size=window_size,
                    index_topk=topk,
                    layer_id=layer_id,
                    n_layers=n_layers,
                    norm_eps=1e-6,
                    rope_yarn=preset["rope_yarn"],
                    indexer_loss_coeff=indexer_loss_coeff,
                ),
                moe=_moe(
                    dim=dim,
                    hidden_dim=preset["moe_inter_dim"],
                    num_experts=preset["num_experts"],
                    num_shared_experts=preset["num_shared_experts"],
                    top_k=preset["top_k"],
                    n_hash_layers=preset["n_hash_layers"],
                    layer_id=layer_id,
                    vocab_size=preset["vocab_size"],
                    route_norm=preset["route_norm"],
                    route_scale=1.5,
                    load_balance_coeff=1e-3,
                    swiglu_limit=swiglu_limit,
                ),
                hc_mult=4,
                sinkhorn_iters=20,
                hc_eps=1e-6,
                norm_eps=1e-6,
            )
        )
    mtp_layers = []
    for depth in range(n_mtp_layers):
        base = layers[-1]
        attn = _attention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_head_dim=32,
            q_lora_rank=preset["q_lora_rank"],
            o_lora_rank=preset["o_lora_rank"],
            n_groups=preset["n_groups"],
            compress_ratio=0,
            window_size=base.attention.window_size,
            index_topk=topk,
            layer_id=n_layers + depth,
            n_layers=n_layers + n_mtp_layers,
            norm_eps=1e-6,
            rope_yarn=False,
            indexer_loss_coeff=0.0,
        )
        mtp_layers.append(
            MTPConfig(
                attention=attn,
                moe=_moe(
                    dim=dim,
                    hidden_dim=preset["moe_inter_dim"],
                    num_experts=preset["num_experts"],
                    num_shared_experts=preset["num_shared_experts"],
                    top_k=preset["top_k"],
                    n_hash_layers=preset["n_hash_layers"],
                    layer_id=n_layers + depth,
                    vocab_size=preset["vocab_size"],
                    route_norm=preset["route_norm"],
                    route_scale=1.5,
                    load_balance_coeff=1e-3,
                    swiglu_limit=swiglu_limit,
                ),
                hc_mult=4,
                sinkhorn_iters=20,
                hc_eps=1e-6,
                norm_eps=1e-6,
            )
        )
    return ModelConfig(
        dim=dim,
        vocab_size=preset["vocab_size"],
        max_context_length=seq_len,
        layers=layers,
        n_mtp_layers=n_mtp_layers,
        mtp_layers=mtp_layers,
        hc_mult=4,
        norm_eps=1e-6,
        tie_word_embeddings=tie_word_embeddings,
    )


MODEL_FLAVORS = (
    "debugmodel",
    "debugmodel_dense",
    "mini_60m",
    "mini_60m_dense",
    "mini_110m",
    "mini_200m",
)


def build_model_config(
    flavor: str,
    *,
    seq_len: int | None = None,
    n_mtp_layers: int = 0,
    tie_word_embeddings: bool | None = None,
    swiglu_limit: float | None = 10.0,
    index_topk: int | None = None,
    indexer_loss_coeff: float = 0.01,
) -> ModelConfig:
    """Build a model config for a named flavor."""
    if flavor not in MODEL_FLAVORS:
        raise ValueError(f"unknown flavor {flavor!r}; available={MODEL_FLAVORS}")
    if flavor.startswith("debugmodel"):
        preset = dict(_DEBUG_PRESET)
        default_len = 16384
        dense = flavor.endswith("_dense")
        tie = False if tie_word_embeddings is None else tie_word_embeddings
    else:
        preset = dict(_MINI_PRESETS[flavor.removesuffix("_dense")])
        default_len = 8192
        dense = flavor.endswith("_dense")
        tie = True if tie_word_embeddings is None else tie_word_embeddings
    context = seq_len or default_len
    if context > default_len:
        raise ValueError(f"seq_len {context} exceeds flavor maximum {default_len}")
    config = _build_preset(
        preset,
        seq_len=context,
        n_mtp_layers=n_mtp_layers,
        dense=dense,
        tie_word_embeddings=tie,
        swiglu_limit=swiglu_limit,
        index_topk=index_topk,
        indexer_loss_coeff=indexer_loss_coeff,
    )
    config.flavor = flavor
    return config
