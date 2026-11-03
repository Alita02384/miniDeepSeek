"""Learning-rate schedules for the single-process trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WarmupDecaySchedule:
    """Warmup followed by cosine/linear/constant decay."""

    optimizer: object
    warmup_steps: int
    total_steps: int
    decay: str = "cosine"
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.decay not in ("cosine", "linear", "constant"):
            raise ValueError(f"unknown decay {self.decay!r}")
        self.last_lr: float | None = None

    def factor(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if self.decay == "constant":
            return 1.0
        # Include the final step in the decay range: with total_steps=10 and
        # warmup=2, step 9 reaches progress 1.0 (the configured minimum).
        decay_steps = max(self.total_steps - self.warmup_steps - 1, 1)
        progress = min(max((step - self.warmup_steps) / decay_steps, 0.0), 1.0)
        if self.decay == "cosine":
            shaped = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:  # linear
            shaped = 1.0 - progress
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * shaped

    def step(self, step: int) -> float:
        factor = self.factor(step)
        for group in self.optimizer.param_groups:
            base_lr = float(group.get("base_lr", group["lr"]))
            group["lr"] = base_lr * factor
        self.last_lr = factor
        return factor
