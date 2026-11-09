"""Auxiliary losses and load-balancing updates.

* sequence-wise balance loss (V4 report section 2.4 / roadmap section 3.1):
  computed from the router gate scores and the hard routing map, per document;
* auxiliary-loss-free expert-bias update (DeepSeek-V3/V4): applied before every
  optimizer step, exactly as torchtitan does.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from ..model.moe import MoE
from ..model.routing import RoutingRecord, iter_routers


@contextlib.contextmanager
def capture_gate_scores(model) -> Iterator[dict[str, Tensor]]:
    """Capture each router's gate projection output during one forward."""
    scores: dict[str, Tensor] = {}
    handles = []
    for name, router in iter_routers(model):
        def hook(module, inputs, output, *, _name=name):
            scores[_name] = output

        handles.append(router.gate.register_forward_hook(hook))
    try:
        yield scores
    finally:
        for handle in handles:
            handle.remove()


def sequence_balance_loss(
    gate_scores: dict[str, Tensor],
    routing: RoutingRecord,
    *,
    num_documents: int,
    cu_seqlens: Tensor,
    top_k: int,
    num_experts: int,
    coeff: float,
) -> Tensor:
    """Sequence-wise load balance loss averaged over layers and documents.

    ``L = coeff * E / k * (1/D) * sum_d sum_i f_{d,i} * P_{d,i}``

    ``f`` is the fraction of tokens in document ``d`` routed to expert ``i`` and
    ``P`` is the mean gate score.  This is the V3/V4 sequence-wise form, which
    discourages a layer from concentrating an individual sequence on a few
    experts (unlike the batch-wise auxiliary loss).  Only learned routers are
    included: hash-routed layers pick experts from a fixed table, so the loss
    cannot change their routing and would only distort the expert weights.
    """
    if coeff <= 0 or not gate_scores or not routing:
        reference = next(iter(gate_scores.values()), None)
        if reference is None:
            return torch.zeros(())
        return reference.new_zeros(())
    if num_experts <= 1:
        # A single expert is a dense FFN: there is no routing decision to
        # balance, and a loss term would only perturb the (fixed) expert weight.
        reference = next(iter(gate_scores.values()))
        return reference.new_zeros(())
    total = None
    layer_count = 0
    boundaries = cu_seqlens
    for name, gate in gate_scores.items():
        if name not in routing.layers:
            continue
        layer = routing.layers[name]
        if layer.is_hash:
            # Hash routing reads a fixed token->expert table, so a balance
            # loss cannot change its routing; its only gradient would distort
            # the gate weights that scale the routed experts.
            continue
        scores = F.softplus(gate.float()).sqrt()
        routing_map = layer.routing_map.to(gate.device).float()
        if scores.shape[0] != routing_map.shape[0]:
            raise RuntimeError(
                f"gate/routing token mismatch for {name}: "
                f"{scores.shape[0]} vs {routing_map.shape[0]}"
            )
        layer_loss = scores.new_zeros(())
        for index in range(boundaries.numel() - 1):
            start = int(boundaries[index])
            end = int(boundaries[index + 1])
            if end <= start:
                continue
            # Drop padding positions (they are not routed).
            mask = torch.ones(end - start, dtype=torch.bool, device=gate.device)
            pair = scores[start:end][mask]
            load = routing_map[start:end][mask]
            if pair.numel() == 0:
                continue
            p_i = pair.mean(dim=0)
            f_i = load.mean(dim=0)
            layer_loss = layer_loss + (f_i * p_i).sum()
        layer_loss = layer_loss / max(1, int(boundaries.numel()) - 1)
        layer_loss = layer_loss * (float(num_experts) / max(int(top_k), 1))
        total = layer_loss if total is None else total + layer_loss
        layer_count += 1
    if total is None or layer_count == 0:
        reference = next(iter(gate_scores.values()))
        return reference.new_zeros(())
    return total / layer_count * float(coeff)


def iter_moe_layers(model):
    """Yield ``(name, MoE)`` for main and MTP MoE layers."""
    for name, module in model.named_modules():
        if isinstance(module, MoE):
            yield name, module


@torch.no_grad()
def update_expert_biases(model) -> int:
    """Auxiliary-loss-free bias update; call once before every optimizer step.

    Only learned routers are steered: a hash layer picks experts from
    ``tid2eid`` and never reads ``expert_bias_E``, so it has no bias at all.
    Counters are reset on every layer, hash or not, because the usage histogram
    is per-step; leaving hash counters uncleared would let the reported expert
    load drift upwards forever.
    """
    moe_layers = list(iter_moe_layers(model))
    updated = 0
    for _name, moe in moe_layers:
        counts = moe.router.tokens_per_expert_E
        if counts.numel() == 0:
            continue
        bias = moe.expert_bias_E
        if not moe.router.hash and bias is not None and moe.load_balance_coeff is not None:
            counts = counts.float()
            delta = moe.load_balance_coeff * torch.sign(counts.mean() - counts)
            delta = delta - delta.mean()
            bias.add_(delta)
            updated += 1
        moe.router.tokens_per_expert_E.zero_()
    return updated


@torch.no_grad()
def reset_expert_usage(model) -> None:
    for _name, moe in iter_moe_layers(model):
        moe.router.tokens_per_expert_E.zero_()


@torch.no_grad()
def router_biases(model) -> dict[str, list[float]]:
    """Current ``expert_bias_E`` for every learned router, keyed by router name.

    The bias is the whole state of the auxiliary-loss-free balancer: it is
    updated every optimizer step and is otherwise invisible in the logs, which
    makes it impossible to tell whether balancing came from the bias update or
    from the auxiliary balance loss.  Hash layers have no bias (they never read
    one), so they are absent rather than reported as zeros.
    """
    output: dict[str, list[float]] = {}
    for name, moe in iter_moe_layers(model):
        bias = getattr(moe, "expert_bias_E", None)
        if bias is None:
            continue
        output[f"{name}.router"] = bias.detach().float().cpu().tolist()
    return output
