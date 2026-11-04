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

import time
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
    routing_metrics,
)
from ..model.wrapper import MiniDeepSeekModel, build_model
from .losses import (
    capture_gate_scores,
    iter_moe_layers,
    reset_expert_usage,
    sequence_balance_loss,
    update_expert_biases,
)
from .metrics import MetricsLogger, swiglu_monitor
from .optimizer import build_optimizer
from .scheduler import IndexTopKSchedule, WarmupDecaySchedule
from .spike import RollbackSnapshot, SpikeDetector, restore, snapshot


@dataclass
class StepMetrics:
    loss: float = 0.0
    main_loss: float = 0.0
    mtp_loss: float = 0.0
    balance_loss: float = 0.0
    tokens: int = 0
    grad_norm: float = 0.0
    lr: float = 0.0
    mtp_weight: float = 0.0
    router: dict[str, float] = field(default_factory=dict)
    swiglu: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "main_loss": self.main_loss,
            "mtp_loss": self.mtp_loss,
            "balance_loss": self.balance_loss,
            "tokens": self.tokens,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "mtp_weight": self.mtp_weight,
            **self.router,
            **self.swiglu,
        }


class Trainer:
    """Train a :class:`MiniDeepSeekModel` on packed batches."""

    def __init__(
        self,
        config: TrainConfig,
        model: MiniDeepSeekModel,
        train_batches: Iterable[PackedBatch] | Callable[[], Iterable[PackedBatch]],
        *,
        val_batches: Iterable[PackedBatch] | None = None,
        tokenizer_hash: str | None = None,
        run_name: str | None = None,
    ):
        self.config = config
        self.model = model
        self.device = model.device
        self.tokenizer_hash = tokenizer_hash
        self.run_name = run_name or Path(config.train.out_dir).name
        set_seed(config.model.seed)
        self.optimizer = build_optimizer(model, config.optim)
        self.scheduler = WarmupDecaySchedule(
            self.optimizer,
            warmup_steps=config.schedule.warmup_steps,
            total_steps=max(config.train.steps, 1),
            decay=config.schedule.decay,
            min_lr_ratio=config.schedule.min_lr_ratio,
        )
        self.logger = MetricsLogger(config.train.out_dir)
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
        self._val_iter = iter(val_batches) if val_batches is not None else None
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
        self._rollback: RollbackSnapshot | None = None
        self._rollback_count = 0
        self._spike_count = 0
        self._use_autocast = config.model.dtype == "bfloat16" and self.device.type == "cuda"
        self._autocast_dtype = torch.bfloat16
        self._first_loss: float | None = None
        self._last_loss: float | None = None

    # -- data -----------------------------------------------------------------
    def _next_batch(self) -> PackedBatch:
        try:
            return next(self._batch_iter)
        except StopIteration:
            self._batch_iter = iter(self._batch_factory())
            return next(self._batch_iter)

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
            target = self._topk_schedule.topk_at(self.step, self.config.train.steps)
            if target != self.model.current_index_topk():
                self.model.set_index_topk(target)

    def _maybe_snapshot(self) -> None:
        every = int(self.config.train.rollback_every)
        if not self.config.train.loss_spike_rollback or every <= 0:
            return
        if self.step % every == 0:
            self._rollback = snapshot(self.step, self.model, self.optimizer)

    def _handle_spike(self, loss: float) -> dict[str, float]:
        is_spike = self.spike_detector.update(loss)
        record: dict[str, float] = {
            "loss_spike": float(is_spike),
            "ar_active": float(self._ar_enabled()),
        }
        if not is_spike:
            return record
        self._spike_count += 1
        if self.config.train.loss_spike_auto_ar and self._ar_state is not None:
            self._ar_force_until = max(
                self._ar_force_until,
                self.step + int(self.config.train.loss_spike_ar_steps),
            )
        if (
            self.config.train.loss_spike_rollback
            and self._rollback is not None
            and self._rollback_count < int(self.config.train.max_rollbacks)
        ):
            restored_step = restore(self._rollback, self.model, self.optimizer)
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
        start = int(self.config.train.steps * self.config.train.mtp_decay_ratio)
        remaining = max(self.config.train.steps - start, 1)
        progress = min(max((self.step - start) / remaining, 0.0), 1.0)
        return initial + (final - initial) * progress

    # -- one microbatch -------------------------------------------------------
    def _forward_backward(self, batch: PackedBatch, mtp_weight: float) -> StepMetrics:
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
            metrics.mtp_loss = (
                float(sum(value.detach() for value in loss_output.mtp.values()) / max(len(loss_output.mtp), 1))
                if loss_output.mtp
                else 0.0
            )
            metrics.loss = float(loss_output.total.detach())
            metrics.balance_loss = float(balance_loss.detach())
            metrics.tokens = loss_output.token_count
            metrics.router = routing_metrics(routing)
            metrics.swiglu = swiglu.as_dict()
            return total, metrics

        total = None
        if self._ar_state is not None and self._ar_enabled():
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
        if not torch.isfinite(total):
            raise RuntimeError(
                f"non-finite loss at step {self.step}: loss={float(total.detach())}"
            )
        (total / accum).backward()
        return metrics

    # -- one training step ----------------------------------------------------
    def train_step(self) -> dict[str, Any]:
        self.model.train()
        self._apply_index_topk()
        self._maybe_snapshot()
        self.optimizer.zero_grad(set_to_none=True)
        reset_expert_usage(self.model.model)
        accum = max(int(self.config.train.grad_accum), 1)
        micro_metrics: list[StepMetrics] = []
        mtp_weight = self._mtp_weight()
        for _ in range(accum):
            batch = self._next_batch()
            micro_metrics.append(self._forward_backward(batch, mtp_weight))
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()), self.config.optim.grad_clip
            )
        )
        update_expert_biases(self.model.model)
        lr_factor = self.scheduler.step(self.step)
        self.optimizer.step()
        if self._ar_state is not None:
            self._ar_state.observe_step(self.model.model)

        averaged = StepMetrics(
            loss=_mean(m.loss + m.balance_loss for m in micro_metrics),
            main_loss=_mean(m.main_loss for m in micro_metrics),
            mtp_loss=_mean(m.mtp_loss for m in micro_metrics),
            balance_loss=_mean(m.balance_loss for m in micro_metrics),
            tokens=sum(m.tokens for m in micro_metrics),
            grad_norm=grad_norm,
            lr=float(self.optimizer.param_groups[0]["lr"]) if self.optimizer.param_groups else 0.0,
            mtp_weight=mtp_weight,
            router=_merge_dicts([m.router for m in micro_metrics]),
            swiglu=_merge_dicts([m.swiglu for m in micro_metrics]),
        )
        if self._first_loss is None:
            self._first_loss = averaged.loss
        self._last_loss = averaged.loss
        record = averaged.as_dict()
        record["index_topk"] = float(self.model.current_index_topk() or 0)
        record["lr_factor"] = lr_factor
        record["lr"] = averaged.lr
        record.update(self._handle_spike(averaged.loss))
        return record

    # -- public loop ----------------------------------------------------------
    def train(self, steps: int | None = None) -> dict[str, Any]:
        total_steps = int(steps if steps is not None else self.config.train.steps)
        started = time.time()
        while self.step < total_steps:
            record = self.train_step()
            if self.step % max(self.config.train.log_every, 1) == 0:
                record["elapsed_s"] = round(time.time() - started, 3)
                self.logger.log(self.step, record)
            if (
                self.config.train.ckpt_every > 0
                and (self.step + 1) % self.config.train.ckpt_every == 0
            ):
                self.save(name=f"step_{self.step + 1:06d}")
            if (
                self.config.train.val_every > 0
                and self._val_iter is not None
                and (self.step + 1) % self.config.train.val_every == 0
            ):
                val = self.evaluate(self._val_iter, max_steps=self.config.train.val_steps)
                self.logger.log(self.step, {"val_loss": val["loss"]}, echo=True)
            self.step += 1
        self.save(name="final")
        summary = self.logger.summary()
        summary.update(
            {
                "steps_run": self.step,
                "first_loss": self._first_loss,
                "last_loss": self._last_loss,
                "wall_seconds": round(time.time() - started, 3),
                "tokens": sum(entry.get("tokens", 0) for entry in self.logger.history),
            }
        )
        return summary

    @torch.no_grad()
    def evaluate(self, batches: Iterable[PackedBatch], *, max_steps: int = 10) -> dict[str, Any]:
        was_training = self.model.training
        self.model.eval()
        losses = []
        tokens = 0
        for index, batch in enumerate(batches):
            if index >= max_steps:
                break
            batch = batch.to(self.device)
            output = self.model.compute_loss(batch, mtp_weight=self._mtp_weight())
            losses.append(float(output.total.detach()))
            tokens += output.token_count
        if was_training:
            self.model.train()
        if not losses:
            return {"loss": float("nan"), "tokens": 0}
        return {"loss": sum(losses) / len(losses), "tokens": tokens}

    # -- checkpoints ----------------------------------------------------------
    def save(self, name: str = "final", *, directory: str | Path | None = None) -> Path:
        path = Path(directory or self.config.train.out_dir) / name
        save_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            step=self.step,
            config=self.config.to_dict(),
            tokenizer_hash=self.tokenizer_hash,
            seed=self.config.model.seed,
            extra={"run_name": self.run_name, "flavor": self.config.model.flavor},
        )
        return path

    def resume(self, directory: str | Path) -> dict[str, Any]:
        meta = load_checkpoint(
            directory, self.model, optimizer=self.optimizer, map_location="cpu"
        )
        self.step = int(meta.get("step", 0))
        return meta

    def close(self) -> dict[str, Any]:
        return self.logger.summary()


def build_trainer(
    config: TrainConfig,
    train_batches: Iterable[PackedBatch] | Callable[[], Iterable[PackedBatch]],
    *,
    val_batches: Iterable[PackedBatch] | None = None,
    tokenizer_hash: str | None = None,
) -> tuple[Trainer, MiniDeepSeekModel]:
    device = resolve_device(config.model.device)
    model = build_model(
        config.model.flavor,
        seq_len=config.model.seq_len,
        n_mtp_layers=config.model.n_mtp_layers,
        device=device,
        tie_word_embeddings=config.model.tie_word_embeddings,
        swiglu_limit=config.model.swiglu_limit,
        index_topk=config.model.index_topk,
        indexer_loss_coeff=config.model.indexer_loss_coeff,
    )
    trainer = Trainer(
        config,
        model,
        train_batches,
        val_batches=val_batches,
        tokenizer_hash=tokenizer_hash,
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
