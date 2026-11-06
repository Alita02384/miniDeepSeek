# Copyright (c) 2026 miniDeepSeek contributors.
# Router/load-balancing design refactored from torchtitan (BSD-3-Clause, Meta).
"""DeepSeek-V4 MoE: hash/learned router, 6 routed experts, 1 shared expert."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import MoEConfig
from .nn import LimitedSwiGLU, depth_scaled_std, init_linear

#: Base seed of the "predefined hash function" (V4 section 2.1).  The table is
#: a pure function of ``(HASH_SEED, layer_id)``: every rank and every process
#: rebuilds exactly the same table, without consuming the global RNG.
HASH_SEED = 0
_LAYER_HASH_STRIDE = 7919


def hash_table_seed(layer_id: int) -> int:
    return (HASH_SEED + layer_id * _LAYER_HASH_STRIDE) % (2**63)


def _build_hash_table(
    vocab: int, experts: int, top_k: int, *, seed: int
) -> torch.Tensor:
    """Build the ``token id -> top-k experts`` table for one hash layer.

    Generation is deterministic: same arguments give the same table on every
    process and device, and it does not advance the global RNG (so parameter
    init is unaffected by the table size).
    """
    if top_k > experts:
        raise ValueError("top_k must be <= num_experts")
    generator = torch.Generator().manual_seed(seed)
    table = torch.empty(vocab, top_k, dtype=torch.long)
    chunk = 8192
    for start in range(0, vocab, chunk):
        stop = min(start + chunk, vocab)
        draws = torch.rand(stop - start, experts, generator=generator)
        table[start:stop] = draws.topk(top_k, dim=-1).indices
    return table


class DeepSeekV4Router(nn.Module):
    """Token-choice router with optional hash routing and loss-free bias."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self.vocab_size = config.vocab_size
        self.n_hash_layers = config.n_hash_layers
        self.layer_id = config.layer_id
        self.hash = config.layer_id < config.n_hash_layers
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        if self.hash:
            self.register_buffer(
                "tid2eid",
                _build_hash_table(
                    config.vocab_size,
                    config.num_experts,
                    config.top_k,
                    seed=hash_table_seed(config.layer_id),
                ),
            )
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def init_weights(self) -> None:
        init_linear(self.gate, depth_scaled_std(0.02, self.layer_id))
        # ``tid2eid`` is data, not an initialized parameter: it is built once in
        # ``__init__`` from ``hash_table_seed(layer_id)`` and afterwards owned by
        # the checkpoint.  Rebuilding it here would silently reroute every hash
        # layer after a load (finite loss, different model), so leave it alone.

    def forward(
        self,
        x: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        del kwargs
        gate = self.gate(x)
        scores = F.softplus(gate.float()).sqrt()
        ids = self._select_experts(
            scores, expert_bias_E, input_ids=input_ids, padding_mask=padding_mask
        )
        topk_scores = scores.gather(-1, ids)
        if self.num_experts == 1:
            # Dense control: a single routed expert must behave exactly like a
            # plain FFN, i.e. weight 1.  A softplus(gate) weight would silently
            # scale the FFN output and make the control non-comparable.
            topk_scores = torch.full_like(topk_scores, float(self.route_scale))
        else:
            if self.route_norm:
                topk_scores = topk_scores / (topk_scores.sum(-1, keepdim=True) + 1e-20)
            topk_scores = topk_scores * self.route_scale
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, ids, True)
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(routing_map.float().sum(0))
        return topk_scores, ids, routing_map

    def _select_experts(
        self,
        scores: torch.Tensor,
        expert_bias_E: torch.Tensor | None,
        *,
        input_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Pick top-k experts; hash layers ignore scores and use the token id."""
        del padding_mask, kwargs
        if self.hash:
            if input_ids is None:
                raise ValueError("hash routing requires input_ids")
            return self.tid2eid.to(input_ids.device)[input_ids.long()]
        choice = scores if expert_bias_E is None else scores + expert_bias_E.float()
        return torch.topk(choice, k=self.top_k, dim=-1, sorted=False).indices


class _ExpertWeight(nn.Module):
    """Named ``weight`` holder so state-dict keys match the reference model."""

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape))


class RoutedExperts(nn.Module):
    """Six small SwiGLU experts; token-choice dispatch with a per-expert loop."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.activation_fn = LimitedSwiGLU(config.swiglu_limit)
        self.layer_id = config.layer_id
        self.w13 = _ExpertWeight((config.num_experts, 2, config.hidden_dim, config.dim))
        self.w2 = _ExpertWeight((config.num_experts, config.dim, config.hidden_dim))

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.w13.weight, std=0.02)
        nn.init.trunc_normal_(
            self.w2.weight, std=depth_scaled_std(0.02, self.layer_id)
        )

    def forward(
        self,
        x: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens, k = topk_ids.shape
        flat_ids = topk_ids.reshape(-1).long()
        flat_scores = topk_scores.reshape(-1)
        owner = torch.arange(tokens, device=x.device).repeat_interleave(k)
        out = torch.zeros_like(x)
        for expert in range(self.num_experts):
            mask = flat_ids == expert
            if not mask.any():
                continue
            token_idx = owner[mask]
            weights = flat_scores[mask]
            x_e = x[token_idx]
            gate_up = torch.einsum("nd,efd->nef", x_e, self.w13.weight[expert])
            hidden = self.activation_fn(gate_up[:, 0], gate_up[:, 1])
            y_e = torch.einsum("nf,df->nd", hidden, self.w2.weight[expert])
            out.index_add_(0, token_idx, y_e * weights.unsqueeze(-1).to(y_e.dtype))
        return out


class SharedExperts(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim * config.num_shared_experts
        self.activation_fn = LimitedSwiGLU(config.swiglu_limit)
        self.layer_id = config.layer_id
        self.w13 = _ExpertWeight((2, self.hidden_dim, config.dim))
        self.w2 = _ExpertWeight((config.dim, self.hidden_dim))

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.w13.weight, std=0.02)
        nn.init.trunc_normal_(
            self.w2.weight, std=depth_scaled_std(0.02, self.layer_id)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.einsum("nd,fd->nf", x, self.w13.weight[0])
        up = torch.einsum("nd,fd->nf", x, self.w13.weight[1])
        hidden = self.activation_fn(gate, up)
        return torch.einsum("nf,df->nd", hidden, self.w2.weight)


class MoE(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.load_balance_coeff = config.load_balance_coeff
        self.router = DeepSeekV4Router(config)
        self.routed_experts = RoutedExperts(config)
        self.shared_experts = (
            SharedExperts(config) if config.num_shared_experts > 0 else None
        )
        # The loss-free bias only steers *learned* routing.  Hash layers select
        # experts from ``tid2eid`` and never read a bias, so they get none: a
        # buffer that is updated but never read is dead checkpoint state.
        if config.load_balance_coeff is not None and not self.router.hash:
            self.register_buffer(
                "expert_bias_E", torch.zeros(config.num_experts, dtype=torch.float32)
            )
        else:
            self.expert_bias_E = None

    def init_weights(self) -> None:
        self.router.init_weights()
        self.routed_experts.init_weights()
        if self.shared_experts is not None:
            self.shared_experts.init_weights()

    def forward(
        self,
        x: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        scores, ids, _ = self.router(
            x, self.expert_bias_E, input_ids=input_ids, padding_mask=padding_mask
        )
        out = self.routed_experts(x, scores, ids)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out
