"""Training metrics: JSONL logging, gradient norms, and SwiGLU clamp stats."""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..model.nn import LimitedSwiGLU


def gradient_norm(model) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum())
    return math.sqrt(total)


@dataclass
class MetricsLogger:
    out_dir: str | Path
    filename: str = "metrics.jsonl"
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / self.filename

    def log(self, step: int, metrics: dict[str, Any], *, echo: bool = True) -> None:
        record = {"step": int(step), **{k: _to_float(v) for k, v in metrics.items()}}
        self.history.append(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if echo:
            print(json.dumps(record, ensure_ascii=False), flush=True)

    def summary(self) -> dict[str, Any]:
        if not self.history:
            return {}
        # Validation records only carry ``val_loss``; intersecting the key sets
        # would drop every training metric as soon as one is logged.
        keys = sorted({key for item in self.history for key in item} - {"step"})
        summary = {"steps": len({item["step"] for item in self.history})}
        for key in keys:
            values = [item[key] for item in self.history if key in item]
            summary[f"{key}_first"] = values[0]
            summary[f"{key}_last"] = values[-1]
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_mean"] = sum(values) / len(values)
        return summary


def _to_float(value: Any) -> Any:
    if isinstance(value, Tensor):
        return float(value.detach().float().mean())
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return value


@dataclass
class SwiGLUStats:
    calls: int = 0
    elements: int = 0
    gate_clamped: int = 0
    up_clamped: int = 0
    gate_abs_max: float = 0.0
    up_abs_max: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "swiglu_calls": float(self.calls),
            "swiglu_gate_clamp_rate": self.gate_clamped / max(self.elements, 1),
            "swiglu_up_clamp_rate": self.up_clamped / max(self.elements, 1),
            "swiglu_gate_abs_max": self.gate_abs_max,
            "swiglu_up_abs_max": self.up_abs_max,
        }


def iter_limited_swiglu(model):
    """Yield every ``LimitedSwiGLU`` activation instance reachable from ``model``."""
    seen = set()
    for module in model.modules():
        activation = getattr(module, "activation_fn", None)
        if isinstance(activation, LimitedSwiGLU) and id(activation) not in seen:
            seen.add(id(activation))
            yield activation


@contextlib.contextmanager
def swiglu_monitor(model) -> Iterator[SwiGLUStats]:
    """Instrument every ``LimitedSwiGLU`` instance for one forward pass.

    ``LimitedSwiGLU`` is a plain callable object, not an ``nn.Module``.  We patch
    the instance's ``_activation_fn`` so the monitor also works when the
    surrounding ``__call__`` is compiled.
    """
    stats = SwiGLUStats()
    patched = []
    for activation in iter_limited_swiglu(model):
        original = activation._activation_fn

        def wrapper(gate: Tensor, up: Tensor, *, _original=original, _activation=activation) -> Tensor:
            with torch.no_grad():
                stats.calls += 1
                stats.elements += gate.numel()
                if _activation.limit is not None:
                    stats.gate_clamped += int((gate > _activation.limit).sum())
                    stats.up_clamped += int((up.abs() > _activation.limit).sum())
                stats.gate_abs_max = max(stats.gate_abs_max, float(gate.detach().abs().max()))
                stats.up_abs_max = max(stats.up_abs_max, float(up.detach().abs().max()))
            return _original(gate, up)

        activation._activation_fn = wrapper
        patched.append((activation, original))
    try:
        yield stats
    finally:
        for activation, original in patched:
            activation._activation_fn = original
