"""Training metrics: JSONL logging, gradient norms, throughput, and SwiGLU stats."""

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

_MIB = 1024.0 * 1024.0


def gradient_norm(model) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum())
    return math.sqrt(total)


def memory_stats(device: torch.device | str) -> dict[str, float]:
    """Return current/peak accelerator memory in MiB, or ``{}`` on CPU.

    Only the CUDA numbers are reliable by construction; MPS exposes current
    allocations but not a peak counter, so ``mem_peak_mb`` is omitted there.
    """
    device = torch.device(device)
    stats: dict[str, float] = {}
    if device.type == "cuda":
        try:
            stats = {
                "mem_allocated_mb": torch.cuda.memory_allocated(device) / _MIB,
                "mem_reserved_mb": torch.cuda.memory_reserved(device) / _MIB,
                "mem_peak_mb": torch.cuda.max_memory_allocated(device) / _MIB,
            }
        except RuntimeError:  # pragma: no cover - driver/context failures
            stats = {}
    elif device.type == "mps" and hasattr(torch, "mps"):  # pragma: no cover - macOS only
        try:
            stats = {"mem_allocated_mb": torch.mps.current_allocated_memory() / _MIB}
        except RuntimeError:
            stats = {}
    return {key: round(float(value), 3) for key, value in stats.items()}


def reset_peak_memory(device: torch.device | str) -> None:
    """Reset the CUDA peak-memory counter so a benchmark window starts clean."""
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def format_count(value: float) -> str:
    """Compact human-readable count: 1.2k, 3.4M, 5.6G."""
    if not math.isfinite(value):
        return "?"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if value >= scale:
            return f"{sign}{value / scale:.2f}{suffix}"
    return f"{sign}{value:.0f}"


def format_duration(seconds: float) -> str:
    """Compact duration: 45s, 12m, 3.4h, 2.1d."""
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def format_metrics(step: int, record: dict[str, Any]) -> str:
    """One compact, human-first line per log record.

    The JSONL file keeps every field for post-hoc analysis; the console line
    only surfaces the values an operator actually watches during a run.
    """
    parts = [f"step {step:>6d}"]
    if (loss := record.get("loss")) is not None:
        text = f"loss {loss:.4f}"
        if (ema := record.get("loss_ema")) is not None:
            text += f" (ema {ema:.4f})"
        parts.append(text)
    if (val := record.get("val_loss")) is not None:
        text = f"val {val:.4f}"
        if (ppl := record.get("val_ppl")) is not None:
            text += f" ppl {ppl:.1f}"
        parts.append(text)
    if (main := record.get("main_loss")) is not None:
        parts.append(f"main {main:.4f}")
    if (mtp := record.get("mtp_loss")) is not None and mtp:
        parts.append(f"mtp {mtp:.4f}")
    if (lr := record.get("lr")) is not None:
        parts.append(f"lr {lr:.2e}")
    if (grad := record.get("grad_norm")) is not None:
        parts.append(f"|g| {grad:.3f}")
    if (rate := record.get("tokens_per_sec")) is not None:
        parts.append(f"{format_count(rate)} tok/s")
    if (mem := record.get("mem_peak_mb")) is not None:
        parts.append(f"{mem / 1024:.1f} GiB peak")
    if (eta := record.get("eta_seconds")) is not None:
        parts.append(f"ETA {format_duration(eta)}")
    return " | ".join(parts)


@dataclass
class MetricsLogger:
    """Append-only JSONL logger for one launch.

    ``run_dir`` is the per-launch directory (``<out_dir>/runs/<run_id>/``) that
    :mod:`minideepseek.runs` allocates.  Without it the logger falls back to a
    flat ``<out_dir>/metrics.jsonl``, which is what older stages and tests use.
    """

    out_dir: str | Path
    filename: str = "metrics.jsonl"
    run_id: str | None = None
    run_dir: str | Path | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = Path(self.run_dir) if self.run_dir is not None else self.out_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.session_dir / self.filename

    def log(self, step: int, metrics: dict[str, Any], *, echo: bool = True) -> None:
        record = {"step": int(step), **{k: _to_float(v) for k, v in metrics.items()}}
        if self.run_id:
            # Kept inside the record as well: it survives anyone concatenating
            # metrics files by hand, which is exactly how the step ranges got
            # mixed up before run directories existed.
            record["run_id"] = self.run_id
        self.history.append(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if echo:
            print(format_metrics(int(step), record), flush=True)

    def summary(self) -> dict[str, Any]:
        if not self.history:
            return {}
        # Validation records only carry ``val_loss``; intersecting the key sets
        # would drop every training metric as soon as one is logged.
        keys = sorted({key for item in self.history for key in item} - {"step"})
        summary = {"steps": len({item["step"] for item in self.history})}
        for key in keys:
            # Some records carry structured payloads (``router_layers`` is a
            # nested dict, ``domain_name`` a string); they have no min/mean.
            values = [
                item[key]
                for item in self.history
                if _is_scalar(item.get(key))
            ]
            if not values:
                continue
            summary[f"{key}_first"] = values[0]
            summary[f"{key}_last"] = values[-1]
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_mean"] = sum(values) / len(values)
        return summary


def _is_scalar(value: Any) -> bool:
    """True for numbers that ``min``/``mean`` can compare (bools excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    #: Device-side accumulators.  The activation hook runs dozens of times per
    #: micro-batch; calling ``.item()`` there cost hundreds of synchronisations
    #: per step, so scalars are materialised once when the monitor exits.
    _counts: Tensor | None = field(default=None, repr=False, compare=False)
    _maxes: Tensor | None = field(default=None, repr=False, compare=False)

    def _accumulate(self, gate: Tensor, up: Tensor, limit: float | None) -> None:
        if limit is not None:
            pair = torch.stack([(gate > limit).sum(), (up.abs() > limit).sum()])
            self._counts = pair if self._counts is None else self._counts + pair
        maxes = torch.stack([gate.detach().abs().max(), up.detach().abs().max()]).float()
        self._maxes = maxes if self._maxes is None else torch.maximum(self._maxes, maxes)

    def _materialize(self) -> None:
        parts = []
        if self._counts is not None:
            parts.append(self._counts.float())
        if self._maxes is not None:
            parts.append(self._maxes)
        if not parts:
            return
        values = torch.cat(parts).tolist()
        if self._counts is not None:
            self.gate_clamped, self.up_clamped = int(values[0]), int(values[1])
            values = values[2:]
        self.gate_abs_max, self.up_abs_max = float(values[0]), float(values[1])

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
                stats._accumulate(gate, up, _activation.limit)
            return _original(gate, up)

        activation._activation_fn = wrapper
        patched.append((activation, original))
    try:
        yield stats
    finally:
        stats._materialize()
        for activation, original in patched:
            activation._activation_fn = original
