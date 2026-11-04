"""Router instrumentation: record, replay, inspect, and anticipate routing.

DeepSeek-V4 uses hash routing for the first ``n_hash_layers`` layers and a
learned router afterwards.  Both expose the same ``topk_expert_ids_TK`` tensor
from ``DeepSeekV4Router.forward``.  Recording those tensors gives us:

* mechanism metrics (expert load, routing entropy) for every training step;
* router replay for GRPO/OPD consistency checks (roadmap section 6);
* the index cache used by Anticipatory Routing (roadmap section 3.6).
"""

from __future__ import annotations

import contextlib
import types
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .moe import DeepSeekV4Router


@dataclass
class LayerRouting:
    name: str
    layer_id: int
    is_hash: bool
    expert_ids: Tensor  # [T, K] int64
    scores: Tensor  # [T, K] float32
    routing_map: Tensor  # [T, E] bool


@dataclass
class RoutingRecord:
    layers: dict[str, LayerRouting] = field(default_factory=dict)
    num_tokens: int = 0

    def __bool__(self) -> bool:
        return bool(self.layers)

    def add(self, name: str, module: DeepSeekV4Router, output) -> None:
        scores, expert_ids, routing_map = output
        if name in self.layers:
            raise RuntimeError(f"router {name!r} was called twice in one recording")
        if self.num_tokens == 0:
            self.num_tokens = int(expert_ids.shape[0])
        elif self.num_tokens != int(expert_ids.shape[0]):
            raise RuntimeError(
                "all routers in a record must see the same token count; "
                f"expected {self.num_tokens}, got {expert_ids.shape[0]} for {name}"
            )
        self.layers[name] = LayerRouting(
            name=name,
            layer_id=int(getattr(module, "layer_id", -1)),
            is_hash=bool(getattr(module, "hash", False)),
            expert_ids=expert_ids.detach().to("cpu", torch.long).clone(),
            scores=scores.detach().float().to("cpu").clone(),
            routing_map=routing_map.detach().to("cpu", torch.bool).clone(),
        )


def iter_routers(model) -> list[tuple[str, DeepSeekV4Router]]:
    """Return ``(qualified_name, router)`` for every router in the model."""
    if not hasattr(model, "named_modules") and hasattr(model, "model"):
        model = model.model
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DeepSeekV4Router)
    ]


@contextlib.contextmanager
def record_routing(model):
    """Record the router decisions of one forward pass.

    Yields a :class:`RoutingRecord` that is filled while the forward runs.
    """
    record = RoutingRecord()
    handles = []
    for name, router in iter_routers(model):
        def hook(module, inputs, output, *, _name=name):
            record.add(_name, module, output)

        handles.append(router.register_forward_hook(hook))
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


def _replay_method(recorded_ids: Tensor):
    def select(this, scores_TE, expert_bias_E=None, **router_kwargs):  # noqa: ANN001
        del expert_bias_E, router_kwargs
        if scores_TE.shape[0] != recorded_ids.shape[0]:
            raise RuntimeError(
                "router replay token count mismatch: "
                f"recorded {recorded_ids.shape[0]}, replay {scores_TE.shape[0]}"
            )
        return recorded_ids.to(device=scores_TE.device, dtype=torch.long)

    return select


@contextlib.contextmanager
def replay_routing(model, record: RoutingRecord):
    """Force every router to return the decisions stored in ``record``.

    Hash routers are replayed too: their table is deterministic, but replaying
    catches any tokenizer/implementation drift between rollout and actor.
    """
    if not record:
        raise ValueError("replay_routing requires a non-empty record")
    patched = []
    try:
        for name, router in iter_routers(model):
            if name not in record.layers:
                raise KeyError(f"record has no routing for layer {name!r}")
            router.__dict__["_select_experts"] = types.MethodType(
                _replay_method(record.layers[name].expert_ids), router
            )
            patched.append(router)
        yield record
    finally:
        for router in patched:
            router.__dict__.pop("_select_experts", None)


def set_routing(record: RoutingRecord, name: str, expert_ids: Tensor) -> None:
    """Replace the recorded expert ids for one layer (used by tests/AR)."""
    layer = record.layers[name]
    if layer.expert_ids.shape != expert_ids.shape:
        raise ValueError(
            f"shape mismatch for {name}: {tuple(layer.expert_ids.shape)} vs "
            f"{tuple(expert_ids.shape)}"
        )
    layer.expert_ids = expert_ids.to(torch.long).clone()


def compare_routing(reference: RoutingRecord, candidate: RoutingRecord) -> dict[str, float]:
    """Set-level top-k agreement rate per layer and overall."""
    if set(reference.layers) != set(candidate.layers):
        raise ValueError("routing records cover different layers")
    per_layer: dict[str, float] = {}
    matches = 0
    total = 0
    for name, ref in reference.layers.items():
        cand = candidate.layers[name]
        if ref.expert_ids.shape != cand.expert_ids.shape:
            raise ValueError(f"routing shape mismatch for {name}")
        ref_sorted = ref.expert_ids.sort(dim=-1).values
        cand_sorted = cand.expert_ids.sort(dim=-1).values
        equal = (ref_sorted == cand_sorted).all(dim=-1)
        per_layer[name] = float(equal.float().mean()) if equal.numel() else 1.0
        matches += int(equal.sum())
        total += int(equal.numel())
    per_layer["overall"] = matches / total if total else 1.0
    return per_layer


def domain_routing_metrics(
    record: RoutingRecord,
    domain_ids: Tensor | list[int],
    *,
    names: dict[int, str] | None = None,
) -> dict[str, float]:
    """Per-domain expert load / entropy / usage spread.

    ``domain_ids[t]`` labels the token at packed position ``t`` (e.g. the
    sample's domain in a GRPO batch).  Returns one group of keys per domain.
    """
    if not record:
        return {}
    ids = torch.as_tensor(domain_ids, dtype=torch.long)
    if ids.numel() != record.num_tokens:
        raise ValueError(
            f"domain_ids has {ids.numel()} entries, record has {record.num_tokens}"
        )
    names = names or {}
    output: dict[str, float] = {}
    for domain in sorted(int(value) for value in ids.unique()):
        label = names.get(domain, str(domain))
        mask = ids == domain
        if not bool(mask.any()):
            continue
        ratios: list[float] = []
        entropies: list[float] = []
        fractions: list[float] = []
        for layer in record.layers.values():
            routing_map = layer.routing_map[mask].float()
            counts = routing_map.sum(dim=0)
            usage = counts / counts.sum().clamp_min(1.0)
            mean = usage.mean().clamp_min(1e-12)
            ratios.append(float(usage.max() / mean))
            entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
            entropies.append(float(entropy / torch.log(torch.tensor(float(usage.numel())))))
            fractions.append(float((counts > 0).float().mean()))
        output[f"domain_{label}_tokens"] = float(mask.sum())
        output[f"domain_{label}_expert_load_max_over_mean"] = sum(ratios) / len(ratios)
        output[f"domain_{label}_routing_entropy_normalized"] = (
            sum(entropies) / len(entropies)
        )
        output[f"domain_{label}_experts_used_fraction"] = (
            sum(fractions) / len(fractions)
        )
    return output


def routing_metrics(record: RoutingRecord) -> dict[str, float]:
    """Expert-load and routing-entropy summaries over all layers."""
    if not record:
        return {}
    max_ratios = []
    entropies = []
    active_fracs = []
    usage_all = []
    for layer in record.layers.values():
        counts = layer.routing_map.float().sum(dim=0)
        usage = counts / counts.sum().clamp_min(1.0)
        mean_count = usage.mean().clamp_min(1e-12)
        max_ratios.append(float(usage.max() / mean_count))
        entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
        entropies.append(float(entropy / torch.log(torch.tensor(float(usage.numel())))))
        active_fracs.append(float((counts > 0).float().mean()))
        usage_all.append(usage)
    usage_mean = torch.stack(usage_all).mean(dim=0)
    return {
        "expert_load_max_over_mean": float(sum(max_ratios) / len(max_ratios)),
        "expert_load_max_over_mean_worst": float(max(max_ratios)),
        "routing_entropy_normalized": float(sum(entropies) / len(entropies)),
        "experts_used_fraction": float(sum(active_fracs) / len(active_fracs)),
        "expert_usage_mean_std": float(usage_mean.std(unbiased=False)),
    }


def router_gate_snapshot(model) -> dict[str, dict[str, Tensor]]:
    """Clone the learned router gate parameters (for Anticipatory Routing)."""
    snapshot: dict[str, dict[str, Tensor]] = {}
    for name, router in iter_routers(model):
        state = {"weight": router.gate.weight.detach().clone()}
        if getattr(router.gate, "bias", None) is not None:
            state["bias"] = router.gate.bias.detach().clone()
        snapshot[name] = state
    return snapshot


@contextlib.contextmanager
def swap_router_gates(model, snapshot: dict[str, dict[str, Tensor]]):
    """Temporarily load router gate weights from ``snapshot``."""
    saved: dict[str, dict[str, Tensor]] = {}
    try:
        for name, router in iter_routers(model):
            if name not in snapshot:
                raise KeyError(f"snapshot missing router {name!r}")
            saved[name] = {"weight": router.gate.weight.detach().clone()}
            if getattr(router.gate, "bias", None) is not None:
                saved[name]["bias"] = router.gate.bias.detach().clone()
            with torch.no_grad():
                router.gate.weight.copy_(snapshot[name]["weight"])
                if "bias" in snapshot[name] and getattr(router.gate, "bias", None) is not None:
                    router.gate.bias.copy_(snapshot[name]["bias"])
        yield
    finally:
        with torch.no_grad():
            for name, router in iter_routers(model):
                if name in saved:
                    router.gate.weight.copy_(saved[name]["weight"])
                    if "bias" in saved[name] and getattr(router.gate, "bias", None) is not None:
                        router.gate.bias.copy_(saved[name]["bias"])
