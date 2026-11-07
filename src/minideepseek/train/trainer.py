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
import re
import shutil
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..config import TrainConfig, resolve_device, set_seed
from ..model.anticipatory import (
    AnticipatoryConfig,
    AnticipatoryState,
    probe_routing,
)
from ..model.batch import PackedBatch
from ..model.checkpoint import load_checkpoint, save_checkpoint
from ..model.routing import (
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
)
from .losses import (
    capture_gate_scores,
    iter_moe_layers,
    reset_expert_usage,
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


class Trainer:
    """Train a :class:`MiniDeepSeekModel` on packed batches."""

    def __init__(
        self,
        config: TrainConfig,
        model: MiniDeepSeekModel,
        train_batches: Iterable[PackedBatch] | Callable[[], Iterable[PackedBatch]],
        *,
        val_batches: Iterable[PackedBatch] | Callable[[], Iterable[PackedBatch]] | None = None,
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
        self.logger = (
            MetricsLogger(config.train.out_dir) if self.dist.is_main else None
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
            )
        self.step = 0
        self._batch_factory = (
            train_batches if callable(train_batches) else (lambda: iter(train_batches))
        )
        self._batch_iter: Iterator[PackedBatch] = iter(self._batch_factory())
        self._batches_consumed = 0
        self._tokens_seen = 0
        self._val_factory: Callable[[], Iterable[PackedBatch]] | None = (
            val_batches
            if callable(val_batches)
            else (lambda: iter(val_batches))
            if val_batches is not None
            else None
        )
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
    ) -> StepMetrics:
        batch = batch.to(self.device)
        accum = max(int(self.config.train.grad_accum), 1)
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
                loss_output = self.model.compute_loss(batch, mtp_weight=mtp_weight)
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
            metrics.main_loss = float(loss_output.main.detach())
            metrics.mtp_losses = {
                int(depth): float(value.detach())
                for depth, value in loss_output.mtp.items()
            }
            metrics.mtp_loss = (
                float(sum(metrics.mtp_losses.values()) / len(metrics.mtp_losses))
                if metrics.mtp_losses
                else 0.0
            )
            metrics.loss = float(loss_output.total.detach())
            metrics.balance_loss = float(balance_loss.detach())
            metrics.indexer_loss = (
                float(loss_output.indexer.detach())
                if loss_output.indexer is not None
                else 0.0
            )
            # ``tokens`` is the LM token budget (main head only); MTP targets
            # are auxiliary prediction slots, not corpus tokens.
            metrics.tokens = loss_output.main_token_count
            metrics.router = routing_metrics(routing)
            metrics.swiglu = swiglu.as_dict()
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
        (total / accum).backward()
        return metrics

    # -- one training step ----------------------------------------------------
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
        for micro in range(accum):
            batch, routing_record = self._next_training_batch()
            with self._gradient_sync(micro == accum - 1):
                micro_metrics.append(
                    self._forward_backward(
                        batch,
                        mtp_weight,
                        routing_record,
                        micro_index=micro,
                    )
                )
        self._tokens_seen += sum(m.tokens for m in micro_metrics)
        # θ_t is still current here; prefetch step t+1's batches with it.
        self._prefetch_step(accum)
        if self.dist.enabled and self.dist.mode == "ddp":
            all_reduce_gradients(self.model.model, self.dist.world_size)
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
        )
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
        averaged.tokens_per_sec = (averaged.tokens * world / elapsed) if elapsed > 0 else 0.0
        averaged.eta_seconds = mean_step * remaining
        self._loss_ema = (
            averaged.loss
            if self._loss_ema is None
            else 0.98 * self._loss_ema + 0.02 * averaged.loss
        )
        averaged.loss_ema = self._loss_ema
        for key, value in memory_stats(self.device).items():
            setattr(averaged, key, value)

        self._last_loss = averaged.loss
        record = averaged.as_dict()
        record["index_topk"] = float(self.model.current_index_topk() or 0)
        record["lr_factor"] = lr_factor
        record.update(self._handle_spike(averaged.loss))
        return record

    # -- public loop ----------------------------------------------------------
    def train(self, steps: int | None = None) -> dict[str, Any]:
        total_steps = int(steps if steps is not None else self.config.train.steps)
        started = time.time()
        while self.step < total_steps:
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
                if self.logger is not None:
                    self.logger.log(self.step, val_record, echo=True)
                self._maybe_save_best(val["loss"])
            if (
                self.config.train.report_every > 0
                and self.step % self.config.train.report_every == 0
            ):
                self._write_report()
            self.step += 1
        self.save(name="final")
        if self.config.train.report_every > 0:
            self._write_report()
        summary = self.logger.summary() if self.logger is not None else {}
        summary.update(
            {
                "steps_run": self.step,
                "first_loss": self._first_loss,
                "last_loss": self._last_loss,
                "wall_seconds": round(time.time() - started, 3),
                "tokens": self._tokens_seen,
                "best_val_loss": (
                    self.best_val_loss if math.isfinite(self.best_val_loss) else None
                ),
                "best_val_step": self._best_val_step,
            }
        )
        return summary

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

    def _write_report(self) -> None:
        """Refresh ``report.html`` from the in-memory metric history."""
        if self.logger is None or not self.logger.history:
            return
        try:
            from ..viz.report import build_report

            build_report(
                self.logger.history,
                Path(self.config.train.out_dir) / "report.html",
                title=f"miniDeepSeek · {self.run_name}",
                meta={
                    "model": self.config.model.flavor,
                    "step": self.step,
                    "tokens_seen": self._tokens_seen,
                    "device": str(self.device),
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
        saved_extra = meta.get("extra", {}) or {}
        best_val = saved_extra.get("best_val_loss")
        if best_val is not None:
            self.best_val_loss = float(best_val)
            self._best_val_step = saved_extra.get("best_val_step")
        # Recreate the deterministic order and skip already-consumed
        # micro-batches; batch-token warmup then continues where it left off.
        consumed = (
            int(saved_extra.get("data_batches", 0))
            if self.config.train.resume_data
            else 0
        )
        while self._batches_consumed < consumed:
            self._next_batch()
        return meta

    def close(self) -> dict[str, Any]:
        return self.logger.summary() if self.logger is not None else {}


def build_trainer(
    config: TrainConfig,
    train_batches: Iterable[PackedBatch] | Callable[[], Iterable[PackedBatch]],
    *,
    val_batches: Iterable[PackedBatch] | Callable[[], Iterable[PackedBatch]] | None = None,
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
        tokenizer_hash=tokenizer_hash,
        distributed=context,
    )
    if config.train.resume:
        trainer.resume(config.train.resume)
    return trainer, model


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _merge_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = set.intersection(*(set(item) for item in dicts)) if dicts else set()
    return {key: _mean(item[key] for item in dicts) for key in keys}


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
