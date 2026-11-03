from __future__ import annotations

import torch

from minideepseek.model.anticipatory import (
    AnticipatoryConfig,
    AnticipatoryState,
    anticipatory_step,
    probe_routing,
)
from minideepseek.model.routing import (
    compare_routing,
    iter_routers,
    record_routing,
    replay_routing,
    router_gate_snapshot,
)


def test_probe_routing_with_stale_gates(debug_model, small_batch):
    snapshot = router_gate_snapshot(debug_model.model)
    with record_routing(debug_model.model) as reference:
        debug_model.forward(small_batch, use_mtp=False)
    # Perturb the live gates: the probe must use the stale snapshot.
    with torch.no_grad():
        for _name, router in iter_routers(debug_model.model):
            router.gate.weight.add_(2.0)
    probe = probe_routing(debug_model, small_batch, snapshot)
    assert compare_routing(reference, probe)["overall"] == 1.0
    with record_routing(debug_model.model) as replayed:
        with replay_routing(debug_model.model, probe):
            debug_model.forward(small_batch, use_mtp=False)
    assert compare_routing(probe, replayed)["overall"] == 1.0


def test_anticipatory_state_ring_buffer(make_debug_model):
    model = make_debug_model()
    state = AnticipatoryState(AnticipatoryConfig(enabled=True, delta=1), step=0)
    assert state.stale_snapshot() is None
    state.observe_step(model.model)
    assert state.stale_snapshot() is not None
    state.observe_step(model.model)
    assert state.stale_snapshot() is not None


def test_anticipatory_step_closure(debug_model, small_batch):
    state = AnticipatoryState(AnticipatoryConfig(enabled=True, delta=1), step=0)
    calls = {"count": 0}

    def closure():
        calls["count"] += 1
        return debug_model.compute_loss(small_batch).total

    # Without a stale snapshot the closure runs once (normal path).
    first = anticipatory_step(debug_model, small_batch, state, forward_closure=closure)
    assert calls["count"] == 1 and torch.isfinite(first)
    state.observe_step(debug_model.model)
    second = anticipatory_step(debug_model, small_batch, state, forward_closure=closure)
    assert calls["count"] == 2 and torch.isfinite(second)
