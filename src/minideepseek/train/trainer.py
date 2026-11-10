"""Single-process trainer for pretraining and SFT.

The loop is intentionally small and explicit: the torchtitan ``Trainer`` assumes
a distributed launcher and its own config registry, while the debug-model stage
needs a readable, CPU-runnable loop that exercises exactly the mechanisms we
care about (hash/learned routing, MTP, SwiGLU clamp, load balancing, checkpoint
round-trips).

Production pretraining can swap in torchtitan's FSDP/EP trainer; the model and
data contracts are the same.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .. import runs
from ..config import TrainConfig, dump_config, resolve_device, set_seed
from ..model.anticipatory import (
    AnticipatoryConfig,
    AnticipatoryState,
    probe_routing,
)
from ..model.batch import PackedBatch
from ..model.checkpoint import load_checkpoint, read_metadata, save_checkpoint
from ..model.routing import (
    domain_expert_specialization,
    layer_expert_counts,
    record_routing,
    replay_routing,
    router_gate_snapshot,
    routing_metrics,
)
from ..model.wrapper import MiniDeepSeekModel, build_model
from .distributed import (
    DistributedContext,
    all_reduce_gradients,
    apply_parallelism,
    clip_grad_norm,
    gather_model_state_dict,
    gather_optimizer_state_dict,
    gradient_sync_disabled,
    init_distributed,
    load_model_state_dict,
    load_optimizer_state_dict,
    scale_gradients,
)
from .losses import (
    capture_gate_scores,
    iter_moe_layers,
    reset_expert_usage,
    router_biases,
    sequence_balance_loss,
    update_expert_biases,
)
from .metrics import (
    MetricsLogger,
    memory_stats,
    swiglu_monitor,
)
from .optimizer import build_optimizer
from .scheduler import IndexTopKSchedule, WarmupDecaySchedule
from .spike import RollbackSnapshot, SpikeDetector, restore, snapshot


@dataclass
class StepMetrics:
    loss: float = 0.0
    main_loss: float = 0.0
    mtp_loss: float = 0.0
    balance_loss: float = 0.0
    indexer_loss: float = 0.0
    tokens: int = 0
    grad_norm: float = 0.0
    lr: float = 0.0
    mtp_weight: float = 0.0
    #: Per-depth MTP losses (``{depth: loss}``); the aggregate ``mtp_loss`` is
    #: still logged so older dashboards keep working.
    mtp_losses: dict[int, float] = field(default_factory=dict)
    #: Cumulative main-head targets consumed by this run.
    tokens_seen: int = 0
    step_time_s: float = 0.0
    tokens_per_sec: float = 0.0
    eta_seconds: float = 0.0
    loss_ema: float = 0.0
    mem_allocated_mb: float | None = None
    mem_reserved_mb: float | None = None
    mem_peak_mb: float | None = None
    router: dict[str, float] = field(default_factory=dict)
    swiglu: dict[str, float] = field(default_factory=dict)
    #: Index of the active ``data.sources`` domain (-1 when single-source).
    #: Under ``data.lockstep`` every micro-batch of a step shares it, so the
    #: per-step routing metrics are already per-domain.
    domain_id: int = -1
    #: Name of the active domain; empty when the loader exposes no mixer.
    domain_name: str = ""
    #: Raw per-router expert counts plus the learned routers' balance bias,
    #: ``{router: {"hash": bool, "counts": [E], "bias": [E]}}``.  Only filled on
    #: snapshot steps (``train.router_every``), since it is the one signal that
    #: cannot be reconstructed from the group-level scalars afterwards.
    router_layers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "loss": self.loss,
            "main_loss": self.main_loss,
            "mtp_loss": self.mtp_loss,
            "balance_loss": self.balance_loss,
            "indexer_loss": self.indexer_loss,
            "tokens": self.tokens,
            "tokens_seen": self.tokens_seen,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "mtp_weight": self.mtp_weight,
            "step_time_s": self.step_time_s,
            "tokens_per_sec": self.tokens_per_sec,
            "eta_seconds": self.eta_seconds,
            "loss_ema": self.loss_ema,
            **self.router,
            **self.swiglu,
        }
        for depth, value in sorted(self.mtp_losses.items()):
            payload[f"mtp_loss_d{depth}"] = value
        # Memory counters are absent on CPU; omit them instead of logging a
        # misleading zero.
        if self.domain_id >= 0:
            payload["domain_id"] = float(self.domain_id)
        if self.domain_name:
            payload["domain_name"] = self.domain_name
        if self.router_layers:
            payload["router_layers"] = self.router_layers
        for key in ("mem_allocated_mb", "mem_reserved_mb", "mem_peak_mb"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


class NonFiniteLossError(RuntimeError):
    """A microbatch produced a NaN/Inf loss; the trainer saves a snapshot."""

    def __init__(
        self,
        message: str,
        *,
        step: int,
        micro: int,
        loss: float,
        metrics: StepMetrics,
    ):
        super().__init__(message)
        self.step = step
        self.micro = micro
        self.loss = loss
        self.metrics = metrics


_CHECKPOINT_STEP_RE = re.compile(r"step_\d+")


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    """Newest resumable checkpoint in ``out_dir`` (``final`` or ``step_*``).

    Compares the recorded step of ``final`` with the numeric ``step_*``
    directories so a stale ``final`` from an earlier run cannot shadow a newer
    periodic checkpoint.
    """
    directory = Path(out_dir)
    if not directory.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    final = directory / "final"
    if final.is_dir():
        try:
            candidates.append((int(read_metadata(final).get("step", 0)), final))
        except (OSError, ValueError, KeyError):
            candidates.append((0, final))
    for child in directory.iterdir():
        if child.is_dir() and _CHECKPOINT_STEP_RE.fullmatch(child.name):
            candidates.append((int(child.name.split("_")[1]), child))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


@contextlib.contextmanager
def _stop_signals(trainer: Trainer):
    """Turn SIGTERM/SIGINT into "finish this step, then checkpoint and exit".

    A second signal re-raises ``KeyboardInterrupt`` so an operator can still
    abort a hung step.  Handlers are only installed in the main thread (the
    Python signal API requirement) and restored afterwards.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, trainer.request_stop)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


#: Loss terms: averaged over ranks weighted by each rank's token count.
_TOKEN_WEIGHTED_KEYS = (
    "loss",
    "main_loss",
    "mtp_loss",
    "mtp_loss_d1",
    "balance_loss",
    "indexer_loss",
    "ar_active",
    "entropy",
    "top1_prob",
)

#: Diagnostic rates/ratios that follow the same token weighting as the losses.
_STAT_PREFIXES = ("router_", "swiglu_")

#: Allocator readings: the maximum across ranks is the binding constraint.
_MEMORY_KEYS = ("mem_allocated_mb", "mem_reserved_mb", "mem_peak_mb")


#: Per-domain routing diagnostics are computed from this rank's own window; a
#: rank whose window happens to miss a domain would not even have the key, and
#: the all-reduce contract needs the same key set on every rank.
_LOCAL_PREFIXES = ("router_domain_",)


def _is_reducible(key: str, value: Any) -> bool:
    """True for a per-rank scalar that should become a whole-job mean."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if key.startswith(_LOCAL_PREFIXES):
        return False
    return key in _TOKEN_WEIGHTED_KEYS or key.startswith(_STAT_PREFIXES)


class Trainer:
    """Train a :class:`MiniDeepSeekModel` on packed batches."""

    def __init__(
        self,
        config: TrainConfig,
        model: MiniDeepSeekModel,
        train_batches: Iterable[PackedBatch] | Callable[[], Iterable[PackedBatch]],
        *,
        val_batches: Iterable[PackedBatch] | Callable[[], Iterable[PackedBatch]] | None = None,
        val_sources: dict[str, Callable[[], Iterable[PackedBatch]]] | None = None,
        tokenizer_hash: str | None = None,
        run_name: str | None = None,
        distributed: DistributedContext | None = None,
    ):
        self.config = config
        self.model = model
        self.device = model.device
        self.dist = distributed or DistributedContext(device=self.device)
        self.tokenizer_hash = tokenizer_hash
        self.run_name = run_name or Path(config.train.out_dir).name
        set_seed(config.model.seed)
        self.optimizer = build_optimizer(model, config.optim)
        schedule_steps = int(config.train.schedule_steps or config.train.steps)
        self.scheduler = WarmupDecaySchedule(
            self.optimizer,
            warmup_steps=config.schedule.warmup_steps,
            total_steps=max(schedule_steps, 1),
            decay=config.schedule.decay,
            min_lr_ratio=config.schedule.min_lr_ratio,
            decay_start_ratio=config.schedule.decay_start_ratio,
        )
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        if self.dist.is_main:
            self._start_run()
        self.logger = (
            MetricsLogger(
                config.train.out_dir,
                run_id=self.run_id,
                run_dir=self.run_dir,
            )
            if self.dist.is_main
            else None
        )
        self._topk_schedule = None
        current_topk = model.current_index_topk()
        if (
            config.train.index_topk_final is not None
            and current_topk is not None
        ):
            self._topk_schedule = IndexTopKSchedule(
                start=int(current_topk),
                final=int(config.train.index_topk_final),
                warmup_ratio=config.train.index_topk_warmup_ratio,
                start_step=config.train.index_topk_start_step,
                end_step=config.train.index_topk_end_step,
            )
        self.step = 0
        self._stop_requested = False
        self._batch_factory = (
            train_batches if callable(train_batches) else (lambda: iter(train_batches))
        )
        self._batch_iter: Iterator[PackedBatch] = iter(self._batch_factory())
        self._batches_consumed = 0
        self._tokens_seen = 0
        # Global (all-rank) tokens pulled from the loader, padding included.
        # This is the data position a resume has to restore: unlike
        # ``_batches_consumed`` it does not change meaning when the GPU count
        # changes (see Trainer._consumed_data_tokens).
        self._data_tokens_global = 0
        #: Predictive-distribution diagnostics of the last metric micro-batch.
        self._entropy_value = 0.0
        self._top1_value = 0.0
        #: Per-domain accumulated expert counts, emitted every ``router_every``.
        self._domain_routing: dict[str, dict[str, Any]] = {}
        self._val_factory: Callable[[], Iterable[PackedBatch]] | None = (
            val_batches
            if callable(val_batches)
            else (lambda: iter(val_batches))
            if val_batches is not None
            else None
        )
        self._val_sources = dict(val_sources or {})
        self._ar_state = (
            AnticipatoryState(
                AnticipatoryConfig(
                    enabled=config.anticipatory.enabled,
                    delta=config.anticipatory.delta,
                    max_steps=config.anticipatory.max_steps,
                )
            )
            if (config.anticipatory.enabled or config.train.loss_spike_auto_ar)
            else None
        )
        self.spike_detector = SpikeDetector(
            window=config.train.loss_spike_window,
            sigma=config.train.loss_spike_sigma,
            min_history=config.train.loss_spike_min_history,
        )
        self._ar_force_until = -1
        # (batch, routing record) computed one step ahead with the current gates.
        self._ar_queue: deque[tuple[PackedBatch, object]] = deque()
        self._rollback: RollbackSnapshot | None = None
        self._rollback_count = 0
        self._spike_count = 0
        self._use_autocast = config.model.dtype == "bfloat16" and self.device.type == "cuda"
        self._autocast_dtype = torch.bfloat16
        self._first_loss: float | None = None
        self._last_loss: float | None = None
        self._loss_ema: float | None = None
        self._step_times: deque[float] = deque(maxlen=50)
        self.best_val_loss: float = float("inf")
        self._best_val_step: int | None = None

    # -- data -----------------------------------------------------------------
    def _pull_batch(self) -> PackedBatch:
        """Fetch the next batch without advancing the resume position."""
        try:
            batch = next(self._batch_iter)
        except StopIteration:
            self._batch_iter = iter(self._batch_factory())
            batch = next(self._batch_iter)
        return batch

    def _next_batch(self) -> PackedBatch:
        batch = self._pull_batch()
        self._batches_consumed += 1
        world = int(self.dist.world_size) if self.dist.enabled else 1
        self._data_tokens_global += int(batch.input_ids.numel()) * world
        return batch

    # -- distributed ----------------------------------------------------------
    @contextlib.contextmanager
    def _gradient_sync(self, enabled: bool):
        if not self.dist.enabled or enabled:
            yield
            return
        inner = self.model.model
        no_sync = getattr(inner, "no_sync", None)
        if no_sync is not None:
            with no_sync():
                yield
        else:
            with gradient_sync_disabled(inner):
                yield

    def _sync_expert_usage(self) -> None:
        """Sum per-rank token counts before the loss-free bias update."""
        if not self.dist.enabled:
            return
        for _name, moe in iter_moe_layers(self.model.model):
            counts = moe.router.tokens_per_expert_E
            if counts.numel():
                self.dist.all_reduce_sum(counts)

    # -- stability ------------------------------------------------------------
    def _ar_enabled(self) -> bool:
        if self._ar_state is None:
            return False
        return (
            self._ar_state.config.enabled
            or self.step < self._ar_force_until
        )

    def _apply_index_topk(self) -> None:
        if self._topk_schedule is not None:
            target = self._topk_schedule.topk_at(
                self.step,
                int(self.config.train.schedule_steps or self.config.train.steps),
            )
            if target != self.model.current_index_topk():
                self.model.set_index_topk(target)

    def _maybe_snapshot(self) -> None:
        every = int(self.config.train.rollback_every)
        if not self.config.train.loss_spike_rollback or every <= 0:
            return
        if self.step % every == 0:
            self._rollback = snapshot(
                self.step,
                self.model.model,
                self.optimizer,
                distributed=self.dist if self.dist.enabled else None,
            )

    def _handle_spike(self, loss: float) -> dict[str, float]:
        is_spike = self.spike_detector.update(loss)
        if self.dist.enabled:
            # All ranks must take the same rollback/AR decision.
            flag = torch.tensor(
                [1.0 if is_spike else 0.0],
                dtype=torch.float32,
                device=self.device,
            )
            torch.distributed.broadcast(flag, src=0)
            is_spike = bool(flag.item())
        record: dict[str, float] = {
            "loss_spike": float(is_spike),
            "ar_active": float(self._ar_enabled()),
        }
        if not is_spike:
            return record
        self._spike_count += 1
        if self.config.train.loss_spike_auto_ar and self._ar_state is not None:
            ar_steps = min(
                int(self.config.train.loss_spike_ar_steps),
                int(self._ar_state.config.max_steps),
            )
            self._ar_force_until = max(
                self._ar_force_until,
                self.step + ar_steps,
            )
        if (
            self.config.train.loss_spike_rollback
            and self._rollback is not None
            and self._rollback_count < int(self.config.train.max_rollbacks)
        ):
            restored_step = restore(
                self._rollback,
                self.model.model,
                self.optimizer,
                distributed=self.dist if self.dist.enabled else None,
            )
            self._ar_queue.clear()
            self._rollback_count += 1
            record["rollback"] = 1.0
            record["rollback_from_step"] = float(self.step)
            record["rollback_to_step"] = float(restored_step)
            self.step = restored_step
        record["ar_active"] = float(self._ar_enabled())
        return record

    # -- schedule -------------------------------------------------------------
    def _mtp_weight(self) -> float:
        initial = float(self.config.train.mtp_weight)
        final = float(
            self.config.train.mtp_weight_final
            if self.config.train.mtp_weight_final is not None
            else initial
        )
        if final == initial:
            return initial
        total_steps = int(self.config.train.schedule_steps or self.config.train.steps)
        start = int(total_steps * self.config.train.mtp_decay_ratio)
        remaining = max(total_steps - start, 1)
        progress = min(max((self.step - start) / remaining, 0.0), 1.0)
        return initial + (final - initial) * progress

    # -- anticipatory prefetch ------------------------------------------------
    def _prefetch_step(self, micro_batches: int) -> None:
        """Fetch one optimizer step's batches and cache θ_t routing.

        This is the V4 §4.2.3 form at optimizer-step granularity: at step t we
        prefetch the ``micro_batches`` batches for t+1 and compute their indices
        with θ_t, so at t+1 they are the stale θ_{t-1} indices.
        """
        if self._ar_state is None or not self._ar_enabled():
            return
        if self._ar_state.config.delta != 1:
            # Prefetching multiple steps ahead needs a gate ring buffer; fall
            # back to the probe path for delta > 1.
            return
        if len(self._ar_queue) >= micro_batches:
            return
        gates = router_gate_snapshot(self.model.model)
        while len(self._ar_queue) < micro_batches:
            # Prefetched batches are not counted as consumed until a training
            # step actually uses them.
            batch = self._pull_batch()
            self._ar_queue.append((batch, probe_routing(self.model, batch, gates)))

    def _next_training_batch(self) -> tuple[PackedBatch, object | None]:
        if self._ar_queue:
            batch, record = self._ar_queue.popleft()
            if self._ar_enabled():
                return batch, record
            # AR was turned off while this batch was queued; use it normally.
            return batch, None
        return self._next_batch(), None

    # -- one microbatch -------------------------------------------------------
    def _forward_backward(
        self,
        batch: PackedBatch,
        mtp_weight: float,
        routing_record=None,
        *,
        micro_index: int = 0,
        collect_metrics: bool = True,
        domain_name: str = "",
    ) -> StepMetrics:
        batch = batch.to(self.device)
        metrics = StepMetrics(mtp_weight=mtp_weight)

        def run_forward() -> tuple[Tensor, StepMetrics]:
            context = (
                torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
                if self._use_autocast
                else _null_context()
            )
            with (
                context, record_routing(self.model.model) as routing,
                capture_gate_scores(self.model.model) as gate_scores,
                swiglu_monitor(self.model.model) as swiglu,
            ):
                loss_output = self.model.compute_loss(
                    batch, mtp_weight=mtp_weight, entropy=collect_metrics
                )
                balance_loss = torch.zeros((), device=self.device)
                if self.config.train.balance_loss_coeff > 0:
                    top_k = 1
                    num_experts = 1
                    for _name, moe in iter_moe_layers(self.model.model):
                        top_k = int(moe.router.top_k)
                        num_experts = int(moe.num_experts)
                        break
                    balance_loss = sequence_balance_loss(
                        gate_scores,
                        routing,
                        num_documents=batch.num_documents,
                        cu_seqlens=batch.cu_seqlens.to(self.device),
                        top_k=top_k,
                        num_experts=num_experts,
                        coeff=self.config.train.balance_loss_coeff,
                    )
                total = loss_output.total + balance_loss
            # One host sync for every scalar of this micro-batch.  Converting
            # them one by one stalled the CPU right after each backward and
            # kept the next micro-batch's forward from overlapping with it.
            mtp_depths = list(loss_output.mtp)
            scalars = [
                loss_output.main.detach(),
                loss_output.total.detach(),
                balance_loss.detach(),
                *[loss_output.mtp[depth].detach() for depth in mtp_depths],
            ]
            if loss_output.indexer is not None:
                scalars.append(loss_output.indexer.detach())
            flat = torch.cat([value.float().reshape(1) for value in scalars]).tolist()
            metrics.main_loss = flat[0]
            metrics.loss = flat[1]
            metrics.balance_loss = flat[2]
            mtp_values = flat[3 : 3 + len(mtp_depths)]
            metrics.mtp_losses = {
                int(depth): value for depth, value in zip(mtp_depths, mtp_values, strict=False)
            }
            metrics.mtp_loss = sum(mtp_values) / len(mtp_values) if mtp_values else 0.0
            metrics.indexer_loss = (
                flat[3 + len(mtp_depths)] if loss_output.indexer is not None else 0.0
            )
            # ``tokens`` is the LM token budget (main head only); MTP targets
            # are auxiliary prediction slots, not corpus tokens.
            metrics.tokens = loss_output.main_token_count
            # Diagnostics cost a reduction + a host sync per router layer per
            # micro-batch (32 of them per optimizer step).  The step-level
            # value is an average anyway, so only the last micro-batch pays for
            # it; ``collect_metrics=False`` keeps the record accumulating but
            # skips the host round-trip.
            if collect_metrics:
                metrics.router = routing_metrics(routing)
                metrics.swiglu = swiglu.as_dict()
                self._entropy_value = float(getattr(loss_output, "entropy", None) or 0.0)
                self._top1_value = float(getattr(loss_output, "top1_prob", None) or 0.0)
                if domain_name:
                    self._accumulate_domain_routing(domain_name, routing)
                if self._router_snapshot_due():
                    metrics.router_layers = layer_expert_counts(routing)
                    metrics.router.update(self._domain_specialization())
            return total, metrics

        total = None
        if routing_record is not None:
            with replay_routing(self.model.model, routing_record):
                total, metrics = run_forward()
        elif self._ar_state is not None and self._ar_enabled():
            stale = self._ar_state.stale_snapshot()
            if stale is not None:
                probe = probe_routing(self.model, batch, stale)
                with replay_routing(self.model.model, probe):
                    total, metrics = run_forward()
            else:
                total, metrics = run_forward()
        else:
            total, metrics = run_forward()

        assert total is not None
        loss_value = float(total.detach())
        if not torch.isfinite(total):
            # Keep the last finite weights on disk: the caller catches this and
            # writes a snapshot before re-raising.
            raise NonFiniteLossError(
                f"non-finite loss at step {self.step} micro {micro_index}: "
                f"loss={loss_value}",
                step=self.step,
                micro=micro_index,
                loss=loss_value,
                metrics=metrics,
            )
        # ``compute_loss`` returns a per-token *mean*, so scaling by the
        # micro-batch's main-head target count turns it into the token-level
        # *sum*.  ``train_step`` divides by the step's total target count after
        # the accumulation loop, making the step gradient the token-weighted
        # mean instead of the mean of per-micro-batch means.  The two differ
        # whenever SFT micro-batches carry different numbers of assistant
        # tokens (they do: the response-only mask is unevenly distributed).
        (total * float(metrics.tokens)).backward()
        return metrics

    # -- one training step ----------------------------------------------------
    def _router_snapshot_due(self) -> bool:
        every = int(self.config.train.router_every)
        return every > 0 and self.step % every == 0

    def _accumulate_domain_routing(self, domain: str, routing) -> None:
        """Add this micro-batch's per-expert counts to the domain's bucket."""
        if not routing:
            return
        bucket = self._domain_routing.setdefault(str(domain), {})
        bucket.setdefault("__batches__", torch.zeros((), device=self.device))
        bucket["__batches__"] = bucket["__batches__"] + 1
        for name, layer in routing.layers.items():
            if layer.is_hash:
                continue  # hash layers cannot specialise by construction
            counts = layer.routing_map.sum(dim=0).float().detach()
            bucket[name] = counts if name not in bucket else bucket[name] + counts

    def _domain_specialization(self) -> dict[str, float]:
        """Per-domain expert concentration, then start a new window."""
        metrics = domain_expert_specialization(
            {name: layers for name, layers in self._domain_routing.items()}
        )
        for name, layers in self._domain_routing.items():
            batches = layers.get("__batches__")
            if batches is not None:
                metrics[f"router_domain_{name}_batches"] = float(batches)
        self._domain_routing = {}
        return metrics

    def _router_layer_snapshot(
        self, micro_metrics: list[StepMetrics]
    ) -> dict[str, dict[str, Any]]:
        """Merge the micro-batch expert counts and attach the current biases.

        Only the last micro-batch of an accumulation group collects routing
        diagnostics, so this normally sees one populated entry; the loop keeps
        the merge correct if that ever changes.
        """
        merged: dict[str, dict[str, Any]] = {}
        for metric in micro_metrics:
            for name, layer in metric.router_layers.items():
                entry = merged.setdefault(
                    name, {"hash": bool(layer.get("hash", False)), "counts": None}
                )
                counts = layer.get("counts") or []
                if entry["counts"] is None:
                    entry["counts"] = [int(value) for value in counts]
                    continue
                for index, value in enumerate(counts):
                    entry["counts"][index] += int(value)
        for name, bias in router_biases(self.model.model).items():
            entry = merged.setdefault(name, {"hash": False, "counts": []})
            entry["bias"] = bias
        return merged

    def train_step(self) -> dict[str, Any]:
        step_started = time.perf_counter()
        self.model.train()
        self._apply_index_topk()
        self._maybe_snapshot()
        self.optimizer.zero_grad(set_to_none=True)
        reset_expert_usage(self.model.model)
        accum = max(int(self.config.train.grad_accum), 1)
        micro_metrics: list[StepMetrics] = []
        mtp_weight = self._mtp_weight()
        step_domain = -1
        step_domain_name = ""
        for micro in range(accum):
            batch, routing_record = self._next_training_batch()
            current = getattr(self._batch_iter, "current_index", None)
            if current is not None:
                step_domain = int(current)
            source = getattr(self._batch_iter, "current_source", None)
            if source is not None:
                step_domain_name = str(source)
            with self._gradient_sync(micro == accum - 1):
                micro_metrics.append(
                    self._forward_backward(
                        batch,
                        mtp_weight,
                        routing_record,
                        micro_index=micro,
                        collect_metrics=micro == accum - 1,
                        domain_name=step_domain_name,
                    )
                )
        self._tokens_seen += sum(m.tokens for m in micro_metrics)
        # θ_t is still current here; prefetch step t+1's batches with it.
        self._prefetch_step(accum)
        if self.dist.enabled and self.dist.mode == "ddp":
            all_reduce_gradients(self.model.model, self.dist.world_size)
        # Gradients currently hold the sum of per-micro-batch token losses.
        # Divide once by the step's target tokens to get the token-weighted
        # mean.  Under equal per-rank token counts (packed batches) this equals
        # the global token mean, matching the DDP average above.
        target_tokens = sum(int(metric.tokens) for metric in micro_metrics)
        if target_tokens > 0:
            scale_gradients(self.model.model, 1.0 / target_tokens)
        grad_norm = clip_grad_norm(self.model.model, self.config.optim.grad_clip)
        self._sync_expert_usage()
        update_expert_biases(self.model.model)
        lr_factor = self.scheduler.step(self.step)
        # Record theta_t *before* the optimizer update: AnticipatoryState
        # labels this snapshot with the completed step and later reads
        # ``step - delta`` as the stale gate set.
        if self._ar_state is not None:
            self._ar_state.observe_step(self.model.model)
        self.optimizer.step()

        mtp_losses: dict[int, float] = {}
        for metric in micro_metrics:
            for depth, value in metric.mtp_losses.items():
                mtp_losses[depth] = mtp_losses.get(depth, 0.0) + value / len(micro_metrics)

        averaged = StepMetrics(
            loss=_mean(m.loss + m.balance_loss for m in micro_metrics),
            main_loss=_mean(m.main_loss for m in micro_metrics),
            mtp_loss=_mean(m.mtp_loss for m in micro_metrics),
            balance_loss=_mean(m.balance_loss for m in micro_metrics),
            indexer_loss=_mean(m.indexer_loss for m in micro_metrics),
            tokens=sum(m.tokens for m in micro_metrics),
            grad_norm=grad_norm,
            lr=float(self.optimizer.param_groups[0]["lr"]) if self.optimizer.param_groups else 0.0,
            mtp_weight=mtp_weight,
            mtp_losses=mtp_losses,
            router=_merge_dicts([m.router for m in micro_metrics]),
            swiglu=_merge_dicts([m.swiglu for m in micro_metrics]),
            domain_id=step_domain,
            domain_name=step_domain_name,
        )
        if self._router_snapshot_due():
            # Read after ``update_expert_biases`` above so the logged bias is
            # the one the next step will actually route with.
            averaged.router_layers = self._router_layer_snapshot(micro_metrics)
        if self._first_loss is None:
            self._first_loss = averaged.loss

        # -- timing / throughput / memory (per optimizer step) ---------------
        elapsed = time.perf_counter() - step_started
        self._step_times.append(elapsed)
        total_steps = int(self.config.train.schedule_steps or self.config.train.steps)
        remaining = max(total_steps - self.step - 1, 0)
        mean_step = sum(self._step_times) / len(self._step_times)
        world = self.dist.world_size if self.dist.enabled else 1
        averaged.tokens_seen = self._tokens_seen
        averaged.step_time_s = elapsed
        # ``averaged.tokens`` is this rank's share; the rate has always been a
        # whole-job figure (x world), which is what the ETA is derived from.
        averaged.tokens_per_sec = (averaged.tokens * world / elapsed) if elapsed > 0 else 0.0
        averaged.eta_seconds = mean_step * remaining
        for key, value in memory_stats(self.device).items():
            setattr(averaged, key, value)

        record = averaged.as_dict()
        record["entropy"] = self._entropy_value
        record["top1_prob"] = self._top1_value
        record = self._globalize(record, tokens=averaged.tokens)
        self._loss_ema = (
            record["loss"]
            if self._loss_ema is None
            else 0.98 * self._loss_ema + 0.02 * record["loss"]
        )
        record["loss_ema"] = self._loss_ema
        self._last_loss = record["loss"]
        record["index_topk"] = float(self.model.current_index_topk() or 0)
        record["lr_factor"] = lr_factor
        record.update(self._handle_spike(record["loss"]))
        return record

    def _globalize(self, record: dict[str, Any], *, tokens: float) -> dict[str, Any]:
        """Rewrite per-rank fields of one metrics record as whole-job figures.

        Without this a 2-GPU log reports half the tokens and a rank-local loss,
        which makes two runs on different GPU counts look like different
        trainings.  Counters are summed, losses and the routing/SwiGLU
        statistics become token-weighted means (every rank processes the same
        number of tokens, so this equals the plain mean), and memory is the
        maximum across ranks -- the binding constraint on the slowest card.
        """
        world = int(self.dist.world_size) if self.dist.enabled else 1
        if world <= 1:
            return record
        record["tokens"] = int(round(tokens * world))
        record["tokens_seen"] = int(round(float(record.get("tokens_seen") or 0.0) * world))
        # Sorted: dict order is insertion order, which differs between ranks
        # (whichever domain showed up first), and an all-reduce over
        # differently-ordered vectors mixes unrelated values.
        weighted = sorted(
            key for key, value in record.items() if _is_reducible(key, value)
        )
        if weighted and tokens > 0:
            values = torch.tensor(
                [float(record[key]) * tokens for key in weighted] + [float(tokens)],
                device=self.device,
                dtype=torch.float64,
            )
            self.dist.all_reduce_sum(values)
            total = float(values[-1])
            if total > 0:
                for index, key in enumerate(weighted):
                    record[key] = float(values[index]) / total
        memory = [key for key in _MEMORY_KEYS if isinstance(record.get(key), (int, float))]
        if memory:
            stats = torch.tensor(
                [float(record[key]) for key in memory],
                device=self.device,
                dtype=torch.float64,
            )
            self.dist.all_reduce_max(stats)
            for index, key in enumerate(memory):
                record[key] = float(stats[index])
        return record

    # -- run identity ---------------------------------------------------------
    def _say(self, message: str) -> None:
        """Print from rank 0 only; every rank logging doubles every line."""
        if self.dist.is_main:
            print(message, flush=True)

    def _start_run(self) -> None:
        """Pick this launch's ``runs/<run_id>/`` and record its provenance.

        A *resumed* training keeps writing into the run directory it already
        had, so one interrupted-and-restarted training stays one metrics file
        (and one line in the report).  A different model, or a run with no
        checkpoint to resume, gets a fresh directory: ``runs/current.json``
        records which run is live, and only a matching flavour with an actual
        checkpoint on disk counts as a continuation.

        The id itself is timestamp + model + world size, so ``ls runs/`` alone
        tells you what ran when.
        """
        report = self.model.parameter_report()
        total_params = float(report.get("total", 0.0))
        world = int(self.dist.world_size) if self.dist.enabled else 1
        flavor = self.config.model.flavor
        continued = runs.continuation(self.config.train.out_dir, flavor)
        if continued:
            self.run_id = continued
            self.run_dir = runs.runs_root(self.config.train.out_dir) / continued
            self.run_dir.mkdir(parents=True, exist_ok=True)
            runs.record_launch(
                self.run_dir,
                world_size=world,
                distributed=bool(self.dist.enabled),
                distributed_mode=str(self.config.distributed.mode),
            )
            dump_config(
                self.config,
                self.run_dir / f"config_{runs.utc_stamp()}.yaml",
            )
            self._say(
                f"[trainer] continuing run {self.run_id} -> {self.run_dir} "
                "(metrics appended)"
            )
            return
        self.run_id = runs.make_run_id(flavor, total_params, world)
        self.run_dir = runs.allocate_run_dir(self.config.train.out_dir, self.run_id)
        self.run_id = self.run_dir.name
        meta = {
            "run_id": self.run_id,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "out_dir": str(self.config.train.out_dir),
            "run_name": self.run_name,
            "model": {
                "flavor": self.config.model.flavor,
                "total_params": total_params,
                "active_params": float(report.get("active_estimate", 0.0)),
                "seq_len": int(self.config.model.seq_len),
                "seed": int(self.config.model.seed),
            },
            "runtime": {
                "device": str(self.device),
                "world_size": world,
                "distributed": bool(self.dist.enabled),
                "distributed_mode": str(self.config.distributed.mode),
                "tokenizer_hash": self.tokenizer_hash,
            },
            "planned_steps": int(self.config.train.steps),
            "git": _git_state(),
        }
        meta["launches"] = [
            {
                "at": meta["started_at"],
                "world_size": world,
                "distributed": bool(self.dist.enabled),
                "distributed_mode": str(self.config.distributed.mode),
            }
        ]
        runs.write_meta(self.run_dir, meta)
        dump_config(self.config, self.run_dir / runs.CONFIG_FILENAME)
        runs.write_current(
            self.config.train.out_dir,
            run_id=self.run_id,
            flavor=flavor,
            total_params=total_params,
        )
        self._say(f"[trainer] run {self.run_id} -> {self.run_dir}")

    def request_stop(self, signum: int | None = None, frame: Any = None) -> None:
        """Ask the loop to checkpoint and exit after the current step."""
        del frame
        if self._stop_requested:
            raise KeyboardInterrupt
        self._stop_requested = True
        name = signal.Signals(signum).name if signum is not None else "stop"
        self._say(
            f"[trainer] {name} received: finishing step {self.step} and saving a "
            "checkpoint; send the signal again to abort immediately"
        )

    # -- public loop ----------------------------------------------------------
    def train(self, steps: int | None = None) -> dict[str, Any]:
        total_steps = int(steps if steps is not None else self.config.train.steps)
        started = time.time()
        with _stop_signals(self):
            self._train_until(total_steps, started)
        summary = self.logger.summary() if self.logger is not None else {}
        summary.update(
            {
                "steps_run": self.step,
                "first_loss": self._first_loss,
                "last_loss": self._last_loss,
                "wall_seconds": round(time.time() - started, 3),
                "tokens": self._tokens_seen
                * (int(self.dist.world_size) if self.dist.enabled else 1),
                "interrupted": bool(self._stop_requested),
                "best_val_loss": (
                    self.best_val_loss if math.isfinite(self.best_val_loss) else None
                ),
                "best_val_step": self._best_val_step,
            }
        )
        return summary

    def _train_until(self, total_steps: int, started: float) -> None:
        while self.step < total_steps:
            if self._stop_requested:
                break
            try:
                record = self.train_step()
            except NonFiniteLossError as error:
                snapshot = self._save_nonfinite_snapshot(error)
                raise NonFiniteLossError(
                    f"{error}\nnon-finite snapshot: {snapshot}",
                    step=error.step,
                    micro=error.micro,
                    loss=error.loss,
                    metrics=error.metrics,
                ) from error
            if (
                self.logger is not None
                and self.step % max(self.config.train.log_every, 1) == 0
            ):
                record["elapsed_s"] = round(time.time() - started, 3)
                self.logger.log(self.step, record)
            if (
                self.config.train.ckpt_every > 0
                and (self.step + 1) % self.config.train.ckpt_every == 0
            ):
                self.save(name=f"step_{self.step + 1:06d}", step=self.step + 1)
                self._cleanup_checkpoints()
            if (
                self.config.train.val_every > 0
                and self._val_factory is not None
                and (self.step + 1) % self.config.train.val_every == 0
            ):
                val = self.evaluate(
                    self._val_factory(), max_steps=self.config.train.val_steps
                )
                val_record = {
                    "val_loss": val["loss"],
                    "val_ppl": val["ppl"],
                    "val_main_loss": val["main_loss"],
                    "val_mtp_loss": val["mtp_loss"],
                    "val_indexer_loss": val["indexer_loss"],
                    "val_tokens": val["tokens"],
                    "tokens_seen": self._tokens_seen,
                }
                # Per-domain validation: one aggregate above, one loss per
                # domain here so general/code/tool regressions are separable.
                for name, factory in self._val_sources.items():
                    per = self.evaluate(
                        factory(), max_steps=self.config.train.val_steps
                    )
                    val_record[f"val_loss_{name}"] = per["loss"]
                    val_record[f"val_main_loss_{name}"] = per["main_loss"]
                    val_record[f"val_tokens_{name}"] = per["tokens"]
                if self.logger is not None:
                    self.logger.log(self.step, val_record, echo=True)
                self._maybe_save_best(val["loss"])
            if (
                self.config.train.report_every > 0
                and self.step % self.config.train.report_every == 0
            ):
                self._write_report()
            self.step += 1
            if self._stop_requested:
                break
        # ``final`` means "this stage finished"; an interrupted run writes a
        # periodic-style checkpoint so the pipeline does not mistake a stopped
        # stage for a completed one.
        self.save(
            name="final" if not self._stop_requested else f"step_{self.step:06d}"
        )
        if self.config.train.report_every > 0:
            self._write_report()

    @torch.no_grad()
    def evaluate(self, batches: Iterable[PackedBatch], *, max_steps: int = 10) -> dict[str, Any]:
        """Token-weighted validation loss plus its components.

        Loss values are weighted by the number of main-head targets so short
        documents cannot dominate the validation number.
        """
        was_training = self.model.training
        self.model.eval()
        totals = {"loss": 0.0, "main": 0.0, "mtp": 0.0, "indexer": 0.0}
        mtp_by_depth: dict[int, float] = {}
        tokens = 0
        mtp_tokens = 0
        batches_seen = 0
        for index, batch in enumerate(batches):
            if index >= max_steps:
                break
            batch = batch.to(self.device)
            output = self.model.compute_loss(batch, mtp_weight=self._mtp_weight())
            count = int(output.main_token_count)
            if count <= 0:
                continue
            batches_seen += 1
            tokens += count
            totals["loss"] += float(output.total.detach()) * count
            totals["main"] += float(output.main.detach()) * count
            if output.mtp:
                mtp_count = max(sum(output.mtp_token_count.values()), 1)
                totals["mtp"] += (
                    sum(float(value.detach()) for value in output.mtp.values())
                    / len(output.mtp)
                    * mtp_count
                )
                mtp_tokens += mtp_count
                for depth, value in output.mtp.items():
                    mtp_by_depth[depth] = mtp_by_depth.get(depth, 0.0) + (
                        float(value.detach()) * mtp_count
                    )
            if output.indexer is not None:
                totals["indexer"] += float(output.indexer.detach()) * count
        if was_training:
            self.model.train()
        if tokens == 0:
            return {
                "loss": float("nan"),
                "main_loss": float("nan"),
                "mtp_loss": float("nan"),
                "indexer_loss": float("nan"),
                "ppl": float("nan"),
                "tokens": 0,
                "batches": 0,
                "mtp_losses": {},
            }
        loss = totals["loss"] / tokens
        main_loss = totals["main"] / tokens
        return {
            "loss": loss,
            # Perplexity is reported on the main LM head only: the training
            # objective also contains MTP/indexer terms, which would make
            # ``exp(loss)`` hard to compare with the literature.
            "ppl": math.exp(min(main_loss, 20.0)),
            "main_loss": main_loss,
            "mtp_loss": totals["mtp"] / max(mtp_tokens, 1),
            "indexer_loss": totals["indexer"] / tokens,
            "tokens": tokens,
            "batches": batches_seen,
            "mtp_losses": {
                depth: value / max(mtp_tokens, 1) for depth, value in mtp_by_depth.items()
            },
        }

    def _report_records(self) -> list[dict[str, Any]]:
        """Metrics for this run plus every run it resumed from.

        Not ``logger.history``: a resumed run starts with an empty in-memory
        history, so rendering that alone would truncate everything logged before
        the interruption.  Not *all* runs either: abandoned launches that
        restarted at step 0 would be spliced into the same curve.
        """
        out_dir = Path(self.config.train.out_dir)
        chain = runs.lineage(out_dir, self.run_id)
        if not chain:
            legacy = runs.legacy_metrics_path(out_dir)
            return runs.load_records(legacy) if legacy is not None else []
        return runs.collect_records(chain, self.logger.filename if self.logger else "metrics.jsonl")

    def _write_report(self) -> None:
        """Refresh ``<out_dir>/report.html`` from this run's lineage."""
        if self.logger is None:
            return
        out_dir = Path(self.config.train.out_dir)
        try:
            from ..viz.report import build_report

            records = self._report_records()
            if not records:
                return
            chain = runs.lineage(out_dir, self.run_id)
            build_report(
                records,
                out_dir / "report.html",
                title=f"miniDeepSeek · {self.run_name}",
                meta={
                    "model": self.config.model.flavor,
                    "step": self.step,
                    "tokens_seen": self._tokens_seen,
                    "device": str(self.device),
                    "run": self.run_id,
                    "runs": len(chain) if chain else 1,
                },
            )
        except Exception as error:  # pragma: no cover - report must never kill training
            print(f"failed to write training report: {error}", flush=True)

    def _maybe_save_best(self, val_loss: float) -> None:
        """Persist ``best`` when validation improves; no-op when disabled."""
        if not self.config.train.save_best or not math.isfinite(val_loss):
            return
        if val_loss >= self.best_val_loss:
            return
        self.best_val_loss = val_loss
        self._best_val_step = self.step + 1
        self.save(
            name="best",
            step=self.step + 1,
            extra={
                "val_loss": val_loss,
                "best_val_loss": self.best_val_loss,
                "best_val_step": self._best_val_step,
            },
        )

    def _save_nonfinite_snapshot(self, error: NonFiniteLossError) -> Path:
        """Write the last finite weights plus a JSON diagnostic before raising.

        The optimizer state is included so a post-mortem can replay the exact
        update history; parameters themselves are still the pre-update values
        because the failure happens during backward.
        """
        name = f"nan_step_{error.step:06d}"
        path = Path(self.config.train.out_dir) / name
        try:
            self.save(
                name=name,
                step=error.step,
                extra={
                    "nonfinite": {
                        "step": error.step,
                        "micro": error.micro,
                        "loss": error.loss,
                    },
                    "best_val_loss": (
                        self.best_val_loss if math.isfinite(self.best_val_loss) else None
                    ),
                },
            )
        except Exception as save_error:  # pragma: no cover - disk/permission failures
            print(f"failed to save non-finite snapshot: {save_error}", flush=True)
        report = {
            "step": error.step,
            "micro": error.micro,
            "loss": error.loss,
            "metrics": error.metrics.as_dict(),
            "recent_history": getattr(self.logger, "history", [])[-10:],
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            path.mkdir(parents=True, exist_ok=True)
            (path / "nan_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except OSError as write_error:  # pragma: no cover
            print(f"failed to write non-finite report: {write_error}", flush=True)
        return path

    def _cleanup_checkpoints(self) -> None:
        """Keep only the most recent ``keep_last_ckpt`` periodic checkpoints."""
        keep = int(self.config.train.keep_last_ckpt)
        if keep <= 0 or not self.dist.is_main:
            return
        out_dir = Path(self.config.train.out_dir)
        if not out_dir.is_dir():
            return
        periodic = sorted(
            (
                child
                for child in out_dir.iterdir()
                if child.is_dir() and re.fullmatch(r"step_\d+", child.name)
            ),
            key=lambda path: int(path.name.split("_")[1]),
        )
        for old in periodic[:-keep]:
            shutil.rmtree(old, ignore_errors=True)

    # -- checkpoints ----------------------------------------------------------
    def save(
        self,
        name: str = "final",
        *,
        directory: str | Path | None = None,
        step: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = Path(directory or self.config.train.out_dir) / name
        save_step = self.step if step is None else int(step)
        payload: dict[str, Any] = {
            "run_name": self.run_name,
            "flavor": self.config.model.flavor,
            "index_topk": self.model.current_index_topk(),
            "data_batches": self._batches_consumed,
            "data_tokens": self._data_tokens_global,
            # Carried across restarts so a resumed run keeps one loss curve:
            # the EMA and the cumulative token counter would otherwise restart
            # from the first post-resume step and put a knee in every plot.
            "loss_ema": self._loss_ema,
            "tokens_seen": self._tokens_seen,
            "best_val_loss": (
                self.best_val_loss if math.isfinite(self.best_val_loss) else None
            ),
        }
        if extra:
            payload.update(extra)
        if self.dist.enabled:
            state = gather_model_state_dict(self.model.model)
            optim_state = gather_optimizer_state_dict(self.model.model, self.optimizer)
            self.dist.barrier()
            if self.dist.is_main:
                save_checkpoint(
                    path,
                    self.model,
                    optimizer=self.optimizer,
                    step=save_step,
                    config=self.config.to_dict(),
                    tokenizer_hash=self.tokenizer_hash,
                    seed=self.config.model.seed,
                    extra=payload,
                    state_dict=state,
                    optimizer_state=optim_state,
                )
            self.dist.barrier()
        else:
            save_checkpoint(
                path,
                self.model,
                optimizer=self.optimizer,
                step=save_step,
                config=self.config.to_dict(),
                tokenizer_hash=self.tokenizer_hash,
                seed=self.config.model.seed,
                extra=payload,
            )
        return path

    def _consumed_data_tokens(
        self, saved_extra: dict[str, Any], meta: dict[str, Any]
    ) -> int:
        """Data offset (in global tokens) a resumed run has to skip.

        The pretrain loader shuffles one *global* pack order and then slices it
        ``indices[rank::world_size]``, so the concatenated pack stream is the
        same for any world size -- only how many packs share a micro-batch
        changes.  A per-rank micro-batch counter therefore means something
        different on 2 GPUs than on 1: replaying 30880 micro-batches of a 2-GPU
        run skips 506M tokens where the single-GPU run had consumed 253M, i.e.
        more than the whole 2K stage.

        Current checkpoints record ``data_tokens`` (global).  For older ones,
        recover the offset from the micro-batch counter and the saved config's
        *global* micro-batch budget; that is exact unless batch-token warmup
        was active (the warmup ratio is 0 in the shipped configs).
        """
        recorded = saved_extra.get("data_tokens")
        if recorded is not None:
            return int(recorded)
        batches = int(saved_extra.get("data_batches", 0) or 0)
        if batches <= 0:
            return 0
        saved_data = (meta.get("config") or {}).get("data") or {}
        per_micro = int(saved_data.get("batch_tokens") or 0)
        if per_micro <= 0:
            raise ValueError(
                "checkpoint has data_batches but neither data_tokens nor "
                "config.data.batch_tokens; cannot recover the data position"
            )
        return batches * per_micro

    def resume(self, directory: str | Path) -> dict[str, Any]:
        if self.dist.enabled:
            import torch

            directory = Path(directory)
            state = torch.load(
                directory / "model.pt", map_location="cpu", weights_only=True
            )
            load_model_state_dict(self.model.model, state)
            optimizer_path = directory / "optimizer.pt"
            if self.config.train.resume_optimizer and optimizer_path.exists():
                optim_state = torch.load(
                    optimizer_path, map_location="cpu", weights_only=True
                )
                load_optimizer_state_dict(self.model.model, self.optimizer, optim_state)
            from ..model.checkpoint import read_metadata

            meta = read_metadata(directory)
        else:
            optimizer = self.optimizer if self.config.train.resume_optimizer else None
            meta = load_checkpoint(
                directory, self.model, optimizer=optimizer, map_location="cpu"
            )
        expected_tokenizer = meta.get("tokenizer_hash")
        if (
            expected_tokenizer
            and self.tokenizer_hash
            and expected_tokenizer != self.tokenizer_hash
        ):
            raise ValueError(
                "checkpoint tokenizer hash mismatch: "
                f"{expected_tokenizer} != {self.tokenizer_hash}"
            )
        self.step = int(meta.get("step", 0)) if self.config.train.resume_step else 0
        if self.run_dir is not None:
            # Provenance: which checkpoint a run started from, so `lineage()`
            # and a human reading meta.json agree on what this run covers.
            runs.update_meta(
                self.run_dir,
                resumed_from=str(directory),
                resumed_at_step=self.step,
            )
        saved_extra = meta.get("extra", {}) or {}
        if self.config.train.resume_optimizer:
            if saved_extra.get("loss_ema") is not None:
                self._loss_ema = float(saved_extra["loss_ema"])
            if saved_extra.get("tokens_seen") is not None:
                self._tokens_seen = int(saved_extra["tokens_seen"])
        best_val = saved_extra.get("best_val_loss")
        if best_val is not None:
            self.best_val_loss = float(best_val)
            self._best_val_step = saved_extra.get("best_val_step")
        # Recreate the deterministic order and skip the already-consumed
        # tokens; batch-token warmup then continues where it left off.  The
        # loop overshoots by at most one micro-batch when the offsets do not
        # line up exactly (batch-token warmup).
        if self.config.train.resume_data:
            consumed = self._consumed_data_tokens(saved_extra, meta)
            while self._data_tokens_global < consumed:
                self._next_batch()
        return meta

    def close(self) -> dict[str, Any]:
        return self.logger.summary() if self.logger is not None else {}


def build_trainer(
    config: TrainConfig,
    train_batches: Iterable[PackedBatch] | Callable[[], Iterable[PackedBatch]],
    *,
    val_batches: Iterable[PackedBatch] | Callable[[], Iterable[PackedBatch]] | None = None,
    val_sources: dict[str, Callable[[], Iterable[PackedBatch]]] | None = None,
    tokenizer_hash: str | None = None,
    distributed: DistributedContext | None = None,
) -> tuple[Trainer, MiniDeepSeekModel]:
    context = distributed or init_distributed(config.distributed, config.model.device)
    device = context.device if context.enabled else resolve_device(config.model.device)
    # Seed before construction: every rank must start from identical weights.
    set_seed(config.model.seed)
    model = build_model(
        config.model,
        device=device,
        dtype=config.model.dtype,
        activation_checkpointing=config.train.activation_checkpointing,
    )
    if context.enabled:
        model.model = apply_parallelism(
            model.model, context, config.distributed, device=device
        )
    trainer = Trainer(
        config,
        model,
        train_batches,
        val_batches=val_batches,
        val_sources=val_sources,
        tokenizer_hash=tokenizer_hash,
        distributed=context,
    )
    resume_path = config.train.resume
    if resume_path is None and config.train.auto_resume:
        found = latest_checkpoint(config.train.out_dir)
        if found is not None:
            resume_path = str(found)
            trainer._say(f"[trainer] auto-resume from {resume_path}")
    if resume_path:
        trainer.resume(resume_path)
    return trainer, model


def _git_state() -> dict[str, Any]:
    """Best-effort ``git rev-parse`` for the provenance record.

    Never fatal: a box can have a git directory owned by another uid (git then
    refuses with "dubious ownership"), or no git at all.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return {
            "commit": commit.stdout.strip() or None,
            "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _merge_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    """Average each key over the micro-batches that actually reported it.

    Intersecting the key sets would drop every router/SwiGLU metric as soon as
    one micro-batch skipped diagnostics: only the last micro-batch of an
    accumulation group collects them (``collect_metrics``), so an intersection
    is empty for any ``grad_accum > 1``.
    """
    merged: dict[str, float] = {}
    for key in {key for item in dicts for key in item}:
        values = [item[key] for item in dicts if key in item]
        merged[key] = sum(values) / len(values)
    return merged


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
