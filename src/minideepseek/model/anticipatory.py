"""Anticipatory Routing (V4 report section 4.2.3).

V4 decouples the routing decision from the current weights: the features are
computed with ``theta_t`` while the router selects experts with ``theta_{t-dt}``.
The indices are prepared one step ahead and cached.

Implementation:

* the trainer prefetches the next optimizer step's batches and caches the
  routing indices computed with the *current* gate weights ``theta_t``;
* the next step replays those cached indices, so it uses ``theta_{t-1}`` while
  its features come from the updated backbone;
* when the queue is empty (first step after enabling, or ``delta > 1``) the
  trainer falls back to a no-grad probe with the stale-gate snapshot.

The probe/prefetch still runs synchronously (no EP communication overlap yet).
The mechanism is implemented and tested but disabled by default (small models have low
spike risk); enable it from the training config with
``anticipatory_routing.enabled=true``.
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
        """Store the router weights that produced step ``self.step``."""
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
            # Include MTP routers so replay can cover the full training forward.
            model.forward(batch)
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
