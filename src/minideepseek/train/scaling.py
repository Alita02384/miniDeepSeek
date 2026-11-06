"""Scaling-law helpers: validation curves and power-law fits.

Training logs store a ``val_loss`` record every ``val_every`` steps.  The
functions here turn those records into ``(tokens, loss)`` points and fit
``L(T) = c · T^α`` in log-log space, the curve a methodology-reproduction
project is ultimately judged on.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationPoint:
    """One validation measurement expressed against tokens seen."""

    run: str
    step: int
    tokens: float
    loss: float


@dataclass
class PowerLawFit:
    """``loss = coefficient * tokens ** alpha`` fitted in log-log space."""

    alpha: float
    coefficient: float
    r2: float
    points: int

    def predict(self, tokens: float) -> float:
        return self.coefficient * float(tokens) ** self.alpha

    def as_dict(self) -> dict[str, float | int]:
        return {
            "alpha": self.alpha,
            "coefficient": self.coefficient,
            "r2": self.r2,
            "points": self.points,
        }


def validation_points(
    records: Iterable[dict],
    *,
    run: str = "run",
    min_tokens: float = 0.0,
    max_tokens: float | None = None,
) -> list[ValidationPoint]:
    """Extract validation points, reconstructing cumulative tokens if needed.

    Older logs only carry per-step ``tokens``; newer ones also log
    ``tokens_seen``.  A backward jump in ``step`` is treated as a new segment
    (curriculum restarts), resetting the accumulator.
    """
    points: list[ValidationPoint] = []
    cumulative = 0.0
    previous_step: int | None = None
    for record in records:
        step = record.get("step")
        if not isinstance(step, (int, float)) or not math.isfinite(float(step)):
            continue
        step_int = int(step)
        if previous_step is not None and step_int < previous_step:
            cumulative = 0.0
        previous_step = step_int
        tokens_step = record.get("tokens")
        if isinstance(tokens_step, (int, float)) and math.isfinite(float(tokens_step)):
            cumulative += float(tokens_step)
        loss = record.get("val_loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
            continue
        tokens_seen = record.get("tokens_seen")
        tokens = (
            float(tokens_seen)
            if isinstance(tokens_seen, (int, float)) and math.isfinite(float(tokens_seen))
            else cumulative
        )
        if tokens <= 0:
            continue
        if tokens < min_tokens:
            continue
        if max_tokens is not None and tokens > max_tokens:
            continue
        points.append(
            ValidationPoint(run=run, step=step_int, tokens=tokens, loss=float(loss))
        )
    return points


def fit_power_law(
    points: Sequence[ValidationPoint],
    *,
    min_tokens: float = 0.0,
    max_tokens: float | None = None,
) -> PowerLawFit:
    """Fit ``log L = α log T + b`` and return α, c = exp(b) and R²."""
    usable = [
        point
        for point in points
        if point.tokens > 0
        and point.loss > 0
        and point.tokens >= min_tokens
        and (max_tokens is None or point.tokens <= max_tokens)
    ]
    if len(usable) < 3:
        raise ValueError(
            f"need at least 3 validation points for a power-law fit, got {len(usable)}"
        )
    x = np.log(np.asarray([point.tokens for point in usable], dtype=np.float64))
    y = np.log(np.asarray([point.loss for point in usable], dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - residual / total if total > 0 else 1.0
    return PowerLawFit(
        alpha=float(slope),
        coefficient=float(math.exp(float(intercept))),
        r2=float(r2),
        points=len(usable),
    )
