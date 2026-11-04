"""Loss-spike detection and (optional) rollback triggers.

V4's stability recipe pairs Anticipatory Routing with a spike detector: when the
loss exceeds a robust threshold over the recent window, roll back briefly,
enable AR for a few steps, then resume normal routing.  This module implements
the detector and the in-memory rollback buffer; both are opt-in because the 60M
debug runs do not need them.
"""

from __future__ import annotations

import copy
import statistics
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SpikeDetector:
    """Robust median/MAD loss-spike detector.

    A loss is a spike when it exceeds ``median + sigma * max(1.4826*MAD,
    relative_floor * median)``.  The relative floor keeps a perfectly flat
    window from flagging tiny numerical noise.
    """

    window: int = 100
    sigma: float = 3.0
    min_history: int = 20
    relative_floor: float = 0.05
    history: deque[float] = field(default_factory=deque)
    spikes: int = 0

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if self.min_history < 2:
            raise ValueError("min_history must be at least 2")
        self.history = deque(maxlen=self.window)

    def threshold(self) -> float | None:
        if len(self.history) < self.min_history:
            return None
        values = list(self.history)
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        scale = max(1.4826 * mad, self.relative_floor * abs(median), 1e-6)
        return median + self.sigma * scale

    def update(self, loss: float) -> bool:
        threshold = self.threshold()
        is_spike = threshold is not None and loss > threshold
        if is_spike:
            self.spikes += 1
        self.history.append(float(loss))
        return is_spike


@dataclass
class RollbackSnapshot:
    step: int
    model_state: dict
    optimizer_state: dict


def snapshot(step: int, model, optimizer, *, distributed=None) -> RollbackSnapshot:
    """Copy model/optimizer state so later updates cannot mutate it.

    With a distributed context the full state is gathered collectively; only
    rank 0 keeps the tensors, other ranks keep empty dicts and participate in
    the later broadcast during :func:`restore`.
    """
    if distributed is not None and distributed.enabled:
        from .distributed import gather_model_state_dict, gather_optimizer_state_dict

        return RollbackSnapshot(
            step=int(step),
            model_state={
                key: value.detach().clone()
                for key, value in gather_model_state_dict(model, all_ranks=True).items()
            },
            optimizer_state=gather_optimizer_state_dict(
                model, optimizer, all_ranks=True
            ),
        )
    return RollbackSnapshot(
        step=int(step),
        model_state=copy.deepcopy(
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
        ),
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
    )


def restore(
    snapshot_state: RollbackSnapshot,
    model,
    optimizer,
    *,
    distributed=None,
) -> int:
    if distributed is not None and distributed.enabled:
        from .distributed import load_model_state_dict, load_optimizer_state_dict

        load_model_state_dict(model, snapshot_state.model_state)
        load_optimizer_state_dict(model, optimizer, snapshot_state.optimizer_state)
        return snapshot_state.step
    model.load_state_dict(
        {key: value.to(next(model.parameters()).device) for key, value in snapshot_state.model_state.items()}
    )
    optimizer.load_state_dict(snapshot_state.optimizer_state)
    return snapshot_state.step
