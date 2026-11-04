from __future__ import annotations

import torch

from minideepseek.model.routing import (
    compare_routing,
    iter_routers,
    record_routing,
    replay_routing,
    router_gate_snapshot,
    routing_metrics,
    swap_router_gates,
)


def test_record_and_replay(debug_model, small_batch):
    with record_routing(debug_model.model) as recorded:
        debug_model.forward(small_batch, use_mtp=False)
    assert len(recorded.layers) == 4
    assert [layer.is_hash for layer in recorded.layers.values()] == [True, True, False, False]
    with record_routing(debug_model.model) as replayed:
        with replay_routing(debug_model.model, recorded):
            debug_model.forward(small_batch, use_mtp=False)
    report = compare_routing(recorded, replayed)
    assert report["overall"] == 1.0


def test_replay_uses_recorded_indices(debug_model, small_batch):
    with record_routing(debug_model.model) as recorded:
        debug_model.forward(small_batch, use_mtp=False)
    name = next(iter(recorded.layers))
    original = recorded.layers[name].expert_ids.clone()
    recorded.layers[name].expert_ids.zero_()
    with record_routing(debug_model.model) as replayed:
        with replay_routing(debug_model.model, recorded):
            debug_model.forward(small_batch, use_mtp=False)
    assert torch.equal(
        replayed.layers[name].expert_ids,
        torch.zeros_like(replayed.layers[name].expert_ids),
    )
    # The forced zeros must differ from what the unmodified router produced.
    assert not torch.equal(replayed.layers[name].expert_ids, original)


def test_routing_metrics(debug_model, small_batch):
    with record_routing(debug_model.model) as recorded:
        debug_model.forward(small_batch, use_mtp=False)
    metrics = routing_metrics(recorded)
    assert metrics["experts_used_fraction"] > 0
    assert 0.0 <= metrics["routing_entropy_normalized"] <= 1.0
    assert metrics["expert_load_max_over_mean"] >= 1.0


def test_gate_snapshot_swap(debug_model, small_batch):
    snapshot = router_gate_snapshot(debug_model.model)
    with record_routing(debug_model.model) as before:
        debug_model.forward(small_batch, use_mtp=False)
    with torch.no_grad():
        for _name, router in iter_routers(debug_model.model):
            bias = torch.arange(router.num_experts, dtype=router.gate.weight.dtype)
            router.gate.weight.add_(bias.unsqueeze(1) * 4.0)
    with swap_router_gates(debug_model.model, snapshot):
        with record_routing(debug_model.model) as after:
            debug_model.forward(small_batch, use_mtp=False)
    assert compare_routing(before, after)["overall"] == 1.0
    # The swap must restore the perturbed weights, which changed the routing.
    with record_routing(debug_model.model) as restored:
        debug_model.forward(small_batch, use_mtp=False)
    assert compare_routing(before, restored)["overall"] < 1.0


def test_hash_router_table_is_persistent(debug_model):
    routers = dict(iter_routers(debug_model.model))
    hash_router = routers["layers.0.feed_forward.router"]
    assert hash_router.hash is True
    assert hash_router.tid2eid.numel() > 0
    buffer_names = dict(debug_model.model.named_buffers())
    assert "layers.0.feed_forward.router.tid2eid" in buffer_names


def test_domain_routing_metrics(debug_model, small_batch):
    from minideepseek.model.routing import domain_routing_metrics

    with record_routing(debug_model.model) as recorded:
        debug_model.forward(small_batch, use_mtp=False)
    domains = [0] * 8 + [1] * 5
    metrics = domain_routing_metrics(recorded, domains, names={0: "general", 1: "code"})
    assert metrics["domain_general_tokens"] == 8
    assert metrics["domain_code_tokens"] == 5
    assert 0.0 <= metrics["domain_general_routing_entropy_normalized"] <= 1.0
    assert metrics["domain_general_expert_load_max_over_mean"] >= 1.0
