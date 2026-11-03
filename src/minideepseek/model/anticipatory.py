"""Anticipatory Routing (V4 report section 4.2.3).

V4 decouples the routing decision from the current weights: the features are
computed with ``theta_t`` while the router selects experts with ``theta_{t-dt}``.
The indices are prepared one step ahead and cached.

Implementation here is the two-pass form described in the roadmap:

1. probe: run a no-grad forward with router gate weights from ``t - delta`` and
   record the resulting top-k indices;
2. train: run the real forward with :func:`replay_routing` so the step uses the
   cached indices, and update the weights with the normal gradient path.

The probe costs one extra forward.  The mechanism is implemented and tested but
disabled by default (60M has low spike risk); enable it from the training config
with ``anticipatory_routing.enabled=true``.
"""

from __future__ import annotations

import contextlib
from collections import deque
from dataclasses import dataclass, field

import torch

from .routing import (
    RoutingRecord,
    record_routing,
    replay_routing,
    router_gate_snapshot,
    swap_router_gates,
)


@dataclass
class AnticipatoryConfig:
    enabled: bool = False
    delta: int = 1
    max_steps: int = 100

    def __post_init__(self) -> None:
        if self.delta < 1:
            raise ValueError("anticipatory routing delta must be >= 1")
        if self.max_steps < 1:
            raise ValueError("anticipatory routing max_steps must be >= 1")


@dataclass
class AnticipatoryState:
    """Stale router-weight ring buffer maintained across optimizer steps."""

    config: AnticipatoryConfig
    snapshots: deque = field(default_factory=lambda: deque(maxlen=8))
    step: int = 0

    def observe_step(self, model) -> None:
        """Store the router weights at the end of a step."""
        self.snapshots.append({"step": self.step, "gates": router_gate_snapshot(model)})
        self.step += 1

    def stale_snapshot(self):
        target = self.step - self.config.delta
        for item in reversed(self.snapshots):
            if item["step"] == target:
                return item["gates"]
        return None


@torch.no_grad()
def probe_routing(model, batch, stale_gates) -> RoutingRecord:
    """Record routing decisions produced by the stale gate weights."""
    was_training = model.training
    model.eval()
    try:
        with swap_router_gates(model, stale_gates), record_routing(model) as record:
            model.forward(batch, use_mtp=False)
    finally:
        if was_training:
            model.train()
    if not record:
        raise RuntimeError("anticipatory routing probe recorded no routers")
    return record


@contextlib.contextmanager
def anticipatory_replay(model, record: RoutingRecord):
    """Replay cached indices during the training forward."""
    with replay_routing(model, record) as replayed:
        yield replayed


def anticipatory_step(
    model,
    batch,
    state: AnticipatoryState,
    *,
    forward_closure,
):
    """Run one Anticipatory-Routing-enabled step.

    ``forward_closure`` performs the actual forward/backward and must return
    whatever the caller needs.  Its forward runs with replayed routing indices.
    """
    if not state.config.enabled:
        return forward_closure()
    stale = state.stale_snapshot()
    if stale is None:
        return forward_closure()
    probe = probe_routing(model, batch, stale)
    with anticipatory_replay(model, probe):
        return forward_closure()
