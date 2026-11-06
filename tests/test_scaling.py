from __future__ import annotations

import math

import pytest

from minideepseek.train.scaling import (
    ValidationPoint,
    fit_power_law,
    validation_points,
)


def _synthetic(alpha: float = -0.05, coefficient: float = 10.0, count: int = 30):
    points = []
    for index in range(count):
        tokens = 1e6 * (1.6**index)
        points.append(
            ValidationPoint(
                run="synth",
                step=index,
                tokens=tokens,
                loss=coefficient * tokens**alpha,
            )
        )
    return points


def test_fit_recovers_known_power_law():
    fit = fit_power_law(_synthetic())
    assert fit.alpha == pytest.approx(-0.05, abs=1e-9)
    assert fit.coefficient == pytest.approx(10.0, rel=1e-9)
    assert fit.r2 == pytest.approx(1.0, abs=1e-9)
    assert fit.predict(1e6) == pytest.approx(10.0 * (1e6**-0.05))


def test_fit_requires_three_points():
    with pytest.raises(ValueError, match="at least 3"):
        fit_power_law(_synthetic(count=2))


def test_validation_points_reconstruct_cumulative_tokens():
    records = [
        {"step": 0, "tokens": 100.0, "loss": 9.0},
        {"step": 0, "val_loss": 8.5},
        {"step": 1, "tokens": 100.0, "loss": 8.0},
        {"step": 1, "val_loss": 7.5},
    ]
    points = validation_points(records)
    assert [(point.step, point.tokens, point.loss) for point in points] == [
        (0, 100.0, 8.5),
        (1, 200.0, 7.5),
    ]


def test_validation_points_prefers_tokens_seen_and_handles_restart():
    records = [
        {"step": 4, "tokens": 100.0, "tokens_seen": 400.0},
        {"step": 4, "val_loss": 6.0, "tokens_seen": 400.0},
        {"step": 0, "tokens": 50.0},
        {"step": 0, "val_loss": 7.0},
    ]
    points = validation_points(records)
    assert [point.tokens for point in points] == [400.0, 50.0]


def test_fit_can_slice_token_range():
    points = _synthetic()
    subset = [point for point in points if point.tokens >= 1e8]
    fit = fit_power_law(points, min_tokens=1e8)
    assert fit.points == len(subset)
    assert math.isclose(fit.alpha, -0.05, abs_tol=1e-9)
