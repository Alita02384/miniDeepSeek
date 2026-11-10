"""Throughput and peak-memory measurement for one optimizer step.

The benchmark deliberately reuses the training path (``compute_loss`` →
backward → clip → optimizer step) so the reported tokens/s is comparable with
real runs.  It is model/device agnostic; the CLI in ``scripts/benchmark.py``
builds the model, batches and (optionally) the distributed wrapper.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from .distributed import clip_grad_norm
from .metrics import memory_stats, reset_peak_memory


@dataclass
class BenchmarkResult:
    """One measured training configuration."""

    label: str = ""
    device: str = ""
    steps: int = 0
    warmup: int = 0
    docs_per_batch: int = 0
    seq_len: int = 0
    tokens_per_step: float = 0.0
    step_time_ms: float = 0.0
    tokens_per_sec: float = 0.0
    peak_mem_mb: float | None = None
    reserved_mem_mb: float | None = None
    params_total: int = 0
    params_active: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def markdown_row(self) -> str:
        peak = f"{self.peak_mem_mb / 1024:.1f} GiB" if self.peak_mem_mb else "—"
        params = f"{self.params_total / 1e6:.1f}M"
        active = f"{self.params_active / 1e6:.1f}M" if self.params_active else "—"
        return (
            f"| {self.label} | {params} | {active} | {self.docs_per_batch} | "
            f"{self.seq_len} | {self.step_time_ms:.0f} | {self.tokens_per_sec:,.0f} | "
            f"{peak} |"
        )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":  # pragma: no cover - macOS only
        torch.mps.synchronize()


def _new_timer(device: torch.device):
    """Return ``(start, stop)`` using CUDA events when available."""
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        def start() -> None:
            start_event.record()

        def stop() -> float:
            end_event.record()
            torch.cuda.synchronize(device)
            return start_event.elapsed_time(end_event) / 1000.0

        return start, stop

    state = {"start": 0.0}

    def start_cpu() -> None:
        state["start"] = time.perf_counter()

    def stop_cpu() -> float:
        return time.perf_counter() - state["start"]

    return start_cpu, stop_cpu


def _one_step(
    model: nn.Module,
    optimizer,
    batch,
    *,
    grad_clip: float,
    mtp_weight: float,
) -> int:
    optimizer.zero_grad(set_to_none=True)
    output = model.compute_loss(batch, mtp_weight=mtp_weight)
    output.total.backward()
    clip_grad_norm(model, grad_clip)
    optimizer.step()
    return int(output.main_token_count)


def synthetic_batch_factory(
    vocab_size: int,
    documents: int,
    seq_len: int,
    device: torch.device,
    *,
    seed: int = 0,
) -> Callable[[], Any]:
    """Random-token packs shaped like a real run (uniform document length).

    Shared by ``scripts/benchmark.py`` and ``scripts/prewarm_attn.py`` so a
    warm-up launches exactly the kernel shapes the trainer will: the packed
    layout, document count and sequence length all feed the launch arguments.
    Tokens stay in ``[3, vocab)`` to avoid the reserved special-token ids.
    """
    from ..model import from_sequences  # local: keeps this module torch-light

    generator = torch.Generator().manual_seed(seed)
    vocab = max(int(vocab_size), 8)

    def factory():
        sequences = [
            torch.randint(3, vocab, (seq_len,), generator=generator).tolist()
            for _ in range(documents)
        ]
        return from_sequences(sequences, device=device)

    return factory


def run_benchmark(
    model: nn.Module,
    optimizer,
    batch_factory: Callable[[], Any],
    *,
    steps: int = 10,
    warmup: int = 3,
    device: torch.device,
    grad_clip: float = 1.0,
    mtp_weight: float = 0.3,
    label: str = "",
) -> BenchmarkResult:
    """Measure tokens/s and peak memory over ``steps`` optimizer steps."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    model.train()
    device = torch.device(device)
    probe = batch_factory()
    docs_per_batch = int(probe.num_documents)
    seq_len = int(probe.num_tokens) // max(docs_per_batch, 1)
    for _ in range(max(warmup, 0)):
        _one_step(
            model,
            optimizer,
            batch_factory(),
            grad_clip=grad_clip,
            mtp_weight=mtp_weight,
        )
    _synchronize(device)
    reset_peak_memory(device)

    timer_start, timer_stop = _new_timer(device)
    timer_start()
    tokens = 0
    for _ in range(steps):
        tokens += _one_step(
            model,
            optimizer,
            batch_factory(),
            grad_clip=grad_clip,
            mtp_weight=mtp_weight,
        )
    elapsed = timer_stop()

    memory = memory_stats(device)
    report = model.parameter_report() if hasattr(model, "parameter_report") else {}
    return BenchmarkResult(
        label=label,
        device=str(device),
        steps=steps,
        warmup=warmup,
        docs_per_batch=docs_per_batch,
        seq_len=seq_len,
        tokens_per_step=tokens / steps,
        step_time_ms=elapsed * 1000.0 / steps,
        tokens_per_sec=tokens / elapsed if elapsed > 0 else 0.0,
        peak_mem_mb=memory.get("mem_peak_mb"),
        reserved_mem_mb=memory.get("mem_reserved_mb"),
        params_total=int(report.get("total", 0)),
        params_active=int(report.get("active_estimate", report.get("active", 0))),
    )
