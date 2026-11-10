"""Learning-rate and sparse-index schedules."""

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
    decay_start_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if not 0.0 <= self.decay_start_ratio <= 1.0:
            raise ValueError("decay_start_ratio must be in [0, 1]")
        if self.decay not in ("cosine", "linear", "constant"):
            raise ValueError(f"unknown decay {self.decay!r}")
        self.last_lr: float | None = None

    def factor(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if self.decay == "constant":
            return 1.0
        # V4 keeps the peak LR until ``decay_start_ratio`` and decays after.
        decay_start = max(
            int(round(self.total_steps * self.decay_start_ratio)),
            self.warmup_steps,
        )
        if step < decay_start:
            return 1.0
        # Include the final step in the decay range: with total_steps=10 and
        # warmup=2, step 9 reaches progress 1.0 (the configured minimum).
        decay_steps = max(self.total_steps - decay_start - 1, 1)
        progress = min(max((step - decay_start) / decay_steps, 0.0), 1.0)
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


@dataclass
class IndexTopKSchedule:
    """Warm the CSA index from dense-over-compressed to sparse top-k.

    Roadmap section 3.5: keep every compressed entry during 4K main training,
    then shrink to ``final`` around the 4K -> 8K transition and stay sparse.

    Two modes:

    - **explicit window** (``start_step``/``end_step``): shrink linearly inside
      that absolute step range, then stay at ``final``.  Curriculum stages use
      this so the shrink happens at the start of the 8K stage instead of at the
      end of the whole run.
    - **legacy ratio**: shrink over the last ``warmup_ratio`` of ``total_steps``.
    """

    start: int
    final: int
    warmup_ratio: float = 0.0
    start_step: int | None = None
    end_step: int | None = None

    def __post_init__(self) -> None:
        if self.start < 1 or self.final < 1:
            raise ValueError("index top-k values must be positive")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if (self.start_step is None) != (self.end_step is None):
            raise ValueError("start_step and end_step must be set together")
        if self.start_step is not None and self.end_step <= self.start_step:
            raise ValueError(
                f"end_step must be greater than start_step, got {self.start_step}..{self.end_step}"
            )

    def _quantise(self, value: float) -> int:
        """Snap to the power-of-two ladder between ``final`` and ``start``.

        Every distinct top-k is a *new Triton specialization*: the gather
        kernel takes ``TOPK`` as a constexpr and statically unrolls the
        selection loop, so the compiler sees a different program per value and
        a cold compile costs minutes (89 s for 64, 28 min for 512 on a 4090).
        Linear interpolation hands the trainer a new integer every step, which
        recompiles the kernel mid-run on every step of the warm-up window.
        Snapping bounds a whole schedule to ``log2(start/final)`` specializations
        that ``scripts/prewarm_attn.py`` can warm ahead of time.
        """
        snapped = 1 << max(0, int(round(math.log2(max(value, 1.0)))))
        return min(max(snapped, self.final), self.start)

    def topk_at(self, step: int, total_steps: int) -> int:
        if self.start_step is not None and self.end_step is not None:
            start, end = self.start_step, self.end_step
            if step <= start:
                return self.start
            if step >= end:
                return self.final
            progress = (step - start) / (end - start)
            value = self.start + (self.final - self.start) * progress
            return self._quantise(value)
        if self.warmup_ratio <= 0 or total_steps <= 0:
            return self.start
        start_step = int(total_steps * (1.0 - self.warmup_ratio))
        if step < start_step:
            return self.start
        span = max(total_steps - start_step, 1)
        progress = min(max((step - start_step) / span, 0.0), 1.0)
        value = self.start + (self.final - self.start) * progress
        return self._quantise(value)
