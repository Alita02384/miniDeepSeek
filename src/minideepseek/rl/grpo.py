"""GRPO trainer (single-process, torch rollout).

Implements the roadmap's correctness-first GRPO recipe:

* group size 8 (configurable), several prompts per step;
* rewards are decomposed and group-normalised into advantages;
* the policy gradient is the clipped PPO/GRPO surrogate;
* rollout and actor log-probs are computed by the *same* model and temperature;
* a router-replay consistency check can be run before training (see
  ``minideepseek.consistency``).

vLLM/verl is the production path.  This trainer exists so the algorithm and the
consistency invariants are testable on CPU with the V4 debug model.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..config import RLConfig, resolve_device, set_seed
from ..data.grpo import GRPORecord
from ..model.checkpoint import load_checkpoint, save_checkpoint
from ..model.routing import domain_routing_metrics, record_routing
from ..model.wrapper import MiniDeepSeekModel, build_model
from ..tokenizer import MiniDeepSeekTokenizer, load_tokenizer
from ..train.distributed import (
    DistributedContext,
    all_reduce_gradients,
    apply_parallelism,
    gather_model_state_dict,
    gather_optimizer_state_dict,
    init_distributed,
    load_model_state_dict,
)
from ..train.losses import iter_moe_layers, update_expert_biases
from ..train.metrics import MetricsLogger
from ..train.optimizer import build_optimizer
from ..train.scheduler import WarmupDecaySchedule
from .agent_rollout import ToolRolloutEngine
from .reward import RewardResult, group_advantages
from .rollout import RolloutGroup, RolloutSample, TorchRolloutEngine


@dataclass
class GRPOStepStats:
    loss: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    advantage_abs_mean: float = 0.0
    response_tokens: int = 0
    response_length_mean: float = 0.0
    clip_fraction: float = 0.0
    ratio_mean: float = 1.0
    approx_kl: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    rollout_weight_version: int = 0
    weight_sync_logprob_diff: float = 0.0
    samples: int = 0
    finished_rate: float = 0.0
    max_new_hit_rate: float = 0.0
    routing: dict[str, float] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "advantage_abs_mean": self.advantage_abs_mean,
            "response_tokens": float(self.response_tokens),
            "response_length_mean": self.response_length_mean,
            "clip_fraction": self.clip_fraction,
            "ratio_mean": self.ratio_mean,
            "approx_kl": self.approx_kl,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "rollout_weight_version": self.rollout_weight_version,
            "weight_sync_logprob_diff": self.weight_sync_logprob_diff,
            "samples": float(self.samples),
            "finished_rate": self.finished_rate,
            "max_new_hit_rate": self.max_new_hit_rate,
            **{f"routing_{key}": value for key, value in self.routing.items()},
            **{f"reward_{key}": value for key, value in self.components.items()},
        }


class GRPOTrainer:
    def __init__(
        self,
        config: RLConfig,
        model: MiniDeepSeekModel,
        tokenizer: MiniDeepSeekTokenizer,
        records: Sequence[GRPORecord],
        reward_fn: Callable[[str, GRPORecord], RewardResult],
        *,
        run_name: str | None = None,
        engine: TorchRolloutEngine | None = None,
        distributed: DistributedContext | None = None,
    ):
        if not records:
            raise ValueError("GRPO requires at least one prompt record")
        if config.rl.group_size < 2:
            raise ValueError("GRPO needs group_size >= 2 to compute advantages")
        _validate_rl_settings(config)
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dist = distributed or DistributedContext(device=model.device)
        self.records = list(records)
        self.reward_fn = reward_fn
        if engine is None:
            if any(record.multi_turn and record.domain == "tool" for record in records):
                engine = ToolRolloutEngine(
                    model,
                    tokenizer,
                    max_turns=config.rl.max_turns,
                    thinking_ratio=config.rl.thinking_ratio,
                    max_tool_calls=config.rl.max_tool_calls,
                )
            else:
                engine = TorchRolloutEngine(
                    model, tokenizer, thinking_ratio=config.rl.thinking_ratio
                )
        self.engine = engine
        self.optimizer = build_optimizer(model, config.optim)
        self.scheduler = WarmupDecaySchedule(
            self.optimizer,
            warmup_steps=config.schedule.warmup_steps,
            total_steps=max(config.train.schedule_steps or config.train.steps, 1),
            decay=config.schedule.decay,
            min_lr_ratio=config.schedule.min_lr_ratio,
            decay_start_ratio=config.schedule.decay_start_ratio,
        )
        self.logger = (
            MetricsLogger(config.train.out_dir) if self.dist.is_main else None
        )
        self.rng = random.Random(config.model.seed)
        self.step_index = 0
        self.run_name = run_name or Path(config.train.out_dir).name
        set_seed(config.model.seed)

    def _sample_records(self) -> list[GRPORecord]:
        # ``prompts_per_step`` is the global (all-rank) prompt budget.  Each
        # rank shards it, so the global number of rollouts per optimizer step
        # stays constant when the DP size changes.
        world = max(int(self.dist.world_size), 1)
        count = min(
            max(int(self.config.rl.prompts_per_step) // world, 1), len(self.records)
        )
        indices = list(range(len(self.records)))
        self.rng.shuffle(indices)
        return [self.records[index] for index in indices[:count]]

    def _flatten(self, groups: list[RolloutGroup]) -> list[RolloutSample]:
        return [sample for group in groups for sample in group.samples]

    def _flatten_advantages(
        self,
        groups: list[RolloutGroup],
        rewards: list[list[RewardResult]],
    ) -> tuple[list[RolloutSample], list[float], list[float]]:
        flat_samples = self._flatten(groups)
        flat_rewards = [result.total for group in rewards for result in group]
        advantages: list[float] = []
        for group_rewards in rewards:
            advantages.extend(group_advantages([result.total for result in group_rewards]))
        if len(advantages) != len(flat_samples):
            raise RuntimeError("advantage/sample accounting mismatch")
        return flat_samples, advantages, flat_rewards

    def _loss_chunk(
        self,
        flat_samples: Sequence[RolloutSample],
        advantages: Sequence[float],
        *,
        total_tokens: int,
        dist_scale: float = 1.0,
    ) -> tuple[Tensor, GRPOStepStats]:
        """Loss on one actor micro-batch, normalised by the *global* token count.

        Dividing by ``total_tokens`` (not the chunk's own count) makes the
        per-chunk backward sum exactly equal to the full-batch mean, which is
        what mini-batching needs.
        """
        batch, response_mask = self.engine.build_batch(flat_samples, self.model.device)
        length = sum(sample.num_loss_tokens for sample in flat_samples)
        if length == 0:
            raise RuntimeError("all sampled responses are empty")
        if self.config.rl.log_domain_routing:
            domain_names: dict[int, str] = {}
            labels: list[int] = []
            for sample in flat_samples:
                domain = sample.record.domain if sample.record else "general"
                index = {name: i for i, name in domain_names.items()}.get(domain)
                if index is None:
                    index = len(domain_names)
                    domain_names[index] = domain
                labels.extend([index] * len(sample.sequence))
            with record_routing(self.model.model) as routing_record:
                actor_logprobs = self.model.response_logprobs(
                    batch,
                    response_mask,
                    requires_grad=True,
                    temperature=self.config.rl.temperature,
                )
            routing_stats = domain_routing_metrics(
                routing_record, labels, names=domain_names
            )
        else:
            actor_logprobs = self.model.response_logprobs(
                batch,
                response_mask,
                requires_grad=True,
                temperature=self.config.rl.temperature,
            )
            routing_stats = {}
        if self.config.rl.recompute_old_logprobs:
            old_logprobs = actor_logprobs.detach()
        else:
            old_logprobs = torch.tensor(
                [value for sample in flat_samples for value in sample.loss_logprobs()],
                dtype=torch.float32,
                device=self.model.device,
            )
        if actor_logprobs.numel() != old_logprobs.numel():
            raise RuntimeError(
                "rollout/actor response token count mismatch: "
                f"{old_logprobs.numel()} vs {actor_logprobs.numel()}"
            )
        advantage_tensor = torch.tensor(
            [
                advantages[index]
                for index, sample in enumerate(flat_samples)
                for _ in range(sample.num_loss_tokens)
            ],
            dtype=torch.float32,
            device=self.model.device,
        )
        # Clamp the log-ratio before exp (DAPO/TitanRL): with a separate
        # generator a stale old logprob must not overflow the ratio to inf.
        log_ratio = (actor_logprobs - old_logprobs).clamp(-10.0, 10.0)
        ratio = torch.exp(log_ratio)
        clip = float(self.config.rl.clip)
        clip_low = float(
            self.config.rl.clip_low if self.config.rl.clip_low is not None else clip
        )
        clip_high = float(
            self.config.rl.clip_high if self.config.rl.clip_high is not None else clip
        )
        if self.config.rl.loss_type == "cispo":
            clamped_ratio = torch.clamp(
                ratio, max=float(self.config.rl.epsilon_high)
            ).detach()
            policy_loss = -(
                clamped_ratio * advantage_tensor * actor_logprobs
            ).sum() / max(total_tokens, 1) * dist_scale
        else:
            unclipped = ratio * advantage_tensor
            clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * advantage_tensor
            policy_loss = (
                -torch.minimum(unclipped, clipped).sum()
                / max(total_tokens, 1)
                * dist_scale
            )
        with torch.no_grad():
            clip_fraction = (
                ((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)).float().mean()
            )
            approx_kl = (old_logprobs - actor_logprobs).mean()
            ratio_mean = ratio.mean()
        stats = GRPOStepStats(
            loss=float(policy_loss.detach()),
            response_tokens=int(length),
            response_length_mean=sum(sample.response_length for sample in flat_samples)
            / len(flat_samples),
            clip_fraction=float(clip_fraction),
            ratio_mean=float(ratio_mean),
            approx_kl=float(approx_kl),
            routing=routing_stats,
        )
        return policy_loss, stats

    def compute_loss(
        self,
        groups: list[RolloutGroup],
        rewards: list[list[RewardResult]],
    ) -> tuple[Tensor, GRPOStepStats]:
        """Full-batch loss (kept for tests and the non-batched path)."""
        flat_samples, advantages, flat_rewards = self._flatten_advantages(groups, rewards)
        total_tokens = sum(sample.num_loss_tokens for sample in flat_samples)
        loss, stats = self._loss_chunk(
            flat_samples, advantages, total_tokens=total_tokens
        )
        stats.samples = len(flat_samples)
        stats.finished_rate = sum(
            1.0 for sample in flat_samples if sample.finished
        ) / len(flat_samples)
        stats.max_new_hit_rate = sum(
            1.0 for sample in flat_samples if sample.max_new_hit
        ) / len(flat_samples)
        stats.reward_mean = sum(flat_rewards) / len(flat_rewards)
        stats.reward_std = _std(flat_rewards)
        stats.advantage_abs_mean = sum(abs(value) for value in advantages) / len(advantages)
        component_values: dict[str, list[float]] = {}
        for group in rewards:
            for result in group:
                for key, value in result.components.items():
                    component_values.setdefault(key, []).append(float(value))
        stats.components = {
            key: sum(values) / len(values) for key, values in component_values.items()
        }
        return loss, stats

    @staticmethod
    def _merge_stats(chunks: list[GRPOStepStats]) -> GRPOStepStats:
        total = sum(stat.response_tokens for stat in chunks)
        if total <= 0:
            raise RuntimeError("no response tokens in GRPO step")

        def weighted(attribute: str) -> float:
            return sum(
                getattr(stat, attribute) * stat.response_tokens for stat in chunks
            ) / total

        routing: dict[str, float] = {}
        for stat in chunks:
            for key, value in stat.routing.items():
                routing[key] = routing.get(key, 0.0) + value * stat.response_tokens
        routing = {key: value / total for key, value in routing.items()}
        lengths = sum(stat.response_length_mean * stat.response_tokens for stat in chunks)
        samples = sum(stat.samples for stat in chunks)
        return GRPOStepStats(
            loss=sum(stat.loss for stat in chunks),
            response_tokens=int(total),
            response_length_mean=lengths / total,
            clip_fraction=weighted("clip_fraction"),
            ratio_mean=weighted("ratio_mean"),
            approx_kl=weighted("approx_kl"),
            samples=samples,
            finished_rate=(
                sum(stat.finished_rate * stat.samples for stat in chunks)
                / max(samples, 1)
            ),
            max_new_hit_rate=(
                sum(stat.max_new_hit_rate * stat.samples for stat in chunks)
                / max(samples, 1)
            ),
            routing=routing,
        )

    def _update_expert_biases(self) -> None:
        # RL also increments the router usage counters in the actor/student
        # forward; settle them before the optimizer step just like the
        # pretrain/SFT trainer does.
        if self.dist.enabled:
            for _name, moe in iter_moe_layers(self.model.model):
                self.dist.all_reduce_sum(moe.router.tokens_per_expert_E)
        update_expert_biases(self.model.model)

    def step(self) -> dict[str, Any]:
        records = self._sample_records()
        groups = self.engine.rollout(
            records,
            group_size=self.config.rl.group_size,
            max_new_tokens=self.config.rl.max_new_tokens,
            temperature=self.config.rl.temperature,
            seed=self.config.model.seed + self.step_index,
        )
        rewards = [
            [
                self.reward_fn(
                    sample.response_text,
                    sample.record or group.record,
                )
                for sample in group.samples
            ]
            for group in groups
        ]
        self.optimizer.zero_grad(set_to_none=True)
        samples = self._flatten(groups)
        sync_diff = 0.0
        every = int(self.config.rl.weight_sync_check_every)
        if every > 0 and self.engine.generator is not None and self.step_index % every == 0:
            diff = self.engine.check_generator_consistency(samples)
            sync_diff = float(diff or 0.0)
            if sync_diff > self.config.rl.weight_sync_threshold:
                raise RuntimeError(
                    f"rollout/actor weight-sync logprob mismatch: {sync_diff:.3e} "
                    f"> {self.config.rl.weight_sync_threshold:.1e}"
                )
        flat_samples, advantages, flat_rewards = self._flatten_advantages(groups, rewards)
        local_tokens = sum(sample.num_loss_tokens for sample in flat_samples)
        if local_tokens == 0:
            raise RuntimeError("all sampled responses are empty")
        if self.dist.enabled:
            count = torch.tensor(
                float(local_tokens), device=self.model.device, dtype=torch.float32
            )
            self.dist.all_reduce_sum(count)
            total_tokens = int(round(float(count.item())))
        else:
            total_tokens = local_tokens
        dist_scale = float(self.dist.world_size) if self.dist.enabled else 1.0
        mini = int(self.config.rl.ppo_mini_batch_size)
        if mini <= 0 or mini >= len(flat_samples):
            chunks = [(flat_samples, advantages)]
        else:
            chunks = [
                (flat_samples[start : start + mini], advantages[start : start + mini])
                for start in range(0, len(flat_samples), mini)
            ]
        chunk_stats: list[GRPOStepStats] = []
        for chunk_samples, chunk_advantages in chunks:
            loss, chunk_stat = self._loss_chunk(
                chunk_samples,
                chunk_advantages,
                total_tokens=total_tokens,
                dist_scale=dist_scale,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite GRPO loss at step {self.step_index}")
            loss.backward()
            chunk_stats.append(chunk_stat)
        stats = self._merge_stats(chunk_stats)
        stats.samples = len(flat_samples)
        stats.finished_rate = sum(
            1.0 for sample in flat_samples if sample.finished
        ) / len(flat_samples)
        stats.max_new_hit_rate = sum(
            1.0 for sample in flat_samples if sample.max_new_hit
        ) / len(flat_samples)
        stats.samples = len(flat_samples)
        stats.finished_rate = sum(
            1.0 for sample in flat_samples if sample.finished
        ) / len(flat_samples)
        stats.max_new_hit_rate = sum(
            1.0 for sample in flat_samples if sample.max_new_hit
        ) / len(flat_samples)
        stats.reward_mean = sum(flat_rewards) / len(flat_rewards)
        stats.reward_std = _std(flat_rewards)
        stats.advantage_abs_mean = sum(abs(value) for value in advantages) / len(advantages)
        component_values: dict[str, list[float]] = {}
        for group in rewards:
            for result in group:
                for key, value in result.components.items():
                    component_values.setdefault(key, []).append(float(value))
        stats.components = {
            key: sum(values) / len(values) for key, values in component_values.items()
        }
        stats.weight_sync_logprob_diff = sync_diff
        if self.dist.enabled and self.dist.mode == "ddp":
            all_reduce_gradients(self.model.model, self.dist.world_size)
        stats.grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()), self.config.optim.grad_clip
            )
        )
        self._update_expert_biases()
        self.scheduler.step(self.step_index)
        self.optimizer.step()
        stats.rollout_weight_version = self.engine.sync_weights(
            self.model.state_dict()
        )
        stats.lr = float(self.optimizer.param_groups[0]["lr"])
        record = stats.as_dict()
        if self.dist.enabled:
            record = _merge_grpo_records(self.dist.all_gather_object(record))
        return record

    def train(self, steps: int | None = None) -> dict[str, Any]:
        total = int(steps if steps is not None else self.config.train.steps)
        while self.step_index < total:
            metrics = self.step()
            if (
                self.logger is not None
                and self.step_index % max(self.config.train.log_every, 1) == 0
            ):
                self.logger.log(self.step_index, metrics)
            if (
                self.config.train.ckpt_every > 0
                and (self.step_index + 1) % self.config.train.ckpt_every == 0
            ):
                self.save(f"step_{self.step_index + 1:06d}", step=self.step_index + 1)
            self.step_index += 1
        self.save("final")
        summary = self.logger.summary() if self.logger is not None else {}
        summary["steps_run"] = self.step_index
        return summary

    def save(self, name: str = "final", *, step: int | None = None) -> Path:
        path = Path(self.config.train.out_dir) / name
        save_step = self.step_index if step is None else int(step)
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
                    tokenizer_hash=self.tokenizer.sha256,
                    seed=self.config.model.seed,
                    extra={
                        "run_name": self.run_name,
                        "method": "grpo",
                        "index_topk": self.model.current_index_topk(),
                    },
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
                tokenizer_hash=self.tokenizer.sha256,
                seed=self.config.model.seed,
                extra={
                    "run_name": self.run_name,
                    "method": "grpo",
                    "index_topk": self.model.current_index_topk(),
                },
            )
        return path

    def resume(self, directory: str | Path) -> dict[str, Any]:
        # D22: RL starts from the model weights only.  Loading the SFT/pretrain
        # optimizer state would also restore its ``base_lr`` and silently
        # override the RL learning rate.
        if not self.dist.enabled:
            meta = load_checkpoint(
                directory,
                self.model,
                optimizer=None,
                map_location="cpu",
                ignore_prefixes=("mtp_layers.",),
            )
        else:
            import torch as _torch

            from ..model.checkpoint import filter_state_dict_prefixes, read_metadata

            path = Path(directory)
            state = _torch.load(path / "model.pt", map_location="cpu", weights_only=True)
            state = filter_state_dict_prefixes(state, ("mtp_layers.",))
            load_model_state_dict(self.model.model, state)
            meta = read_metadata(path)
        self.step_index = (
            int(meta.get("step", 0)) if self.config.train.resume_step else 0
        )
        return meta


def build_grpo_trainer(
    config: RLConfig,
    records: Sequence[GRPORecord],
    reward_fn: Callable[[str, GRPORecord], RewardResult],
    *,
    tokenizer: MiniDeepSeekTokenizer | None = None,
    distributed: DistributedContext | None = None,
) -> tuple[GRPOTrainer, MiniDeepSeekModel]:
    context = distributed or init_distributed(config.distributed, config.model.device)
    if (
        config.rl.rollout_backend in {"cached", "kv"}
        and context.enabled
        and context.mode != "ddp"
    ):
        raise ValueError(
            "cached rollout keeps one full model replica per rank; use "
            "distributed.mode=ddp (D81). FSDP2 needs a gathered rollout copy, "
            "which is not wired yet."
        )
    device = context.device if context.enabled else resolve_device(config.model.device)
    set_seed(config.model.seed)
    model = build_model(
        config.model,
        device=device,
        dtype=config.model.dtype,
        activation_checkpointing=bool(config.train.activation_checkpointing),
    )
    _disable_indexer_aux(model)
    if context.enabled:
        model.model = apply_parallelism(
            model.model, context, config.distributed, device=device
        )
        if config.rl.rollout_backend == "vllm":
            raise ValueError(
                "vLLM rollout with FSDP2 GRPO is not wired; use the torch backend "
                "or run verl's FSDP actor."
            )
        records = list(records)[context.rank :: context.world_size]
        if not records:
            raise ValueError(
                f"rank {context.rank} has no GRPO records after sharding"
            )
    tokenizer = tokenizer or load_tokenizer(vocab_limit=config.model.vocab_limit)
    from .backends import build_rollout_engine

    engine = build_rollout_engine(config, model, tokenizer)
    tool_records = any(
        record.multi_turn and record.domain == "tool" for record in records
    )
    if tool_records:
        if config.rl.rollout_backend in {"cached", "kv"}:
            from .cached_tool import CachedToolRolloutEngine

            engine = CachedToolRolloutEngine(
                model,
                tokenizer,
                max_turns=config.rl.max_turns,
                thinking_ratio=config.rl.thinking_ratio,
                max_tool_calls=config.rl.max_tool_calls,
                max_context=int(config.model.seq_len),
            )
        else:
            engine = ToolRolloutEngine(
                model,
                tokenizer,
                generator=getattr(engine, "generator", None),
                max_turns=config.rl.max_turns,
                thinking_ratio=config.rl.thinking_ratio,
                max_tool_calls=config.rl.max_tool_calls,
            )
    trainer = GRPOTrainer(
        config,
        model,
        tokenizer,
        records,
        reward_fn,
        engine=engine,
        distributed=context,
    )
    if config.train.resume:
        meta = trainer.resume(config.train.resume)
        expected = meta.get("tokenizer_hash")
        if expected and expected != tokenizer.sha256:
            raise ValueError("GRPO resume checkpoint tokenizer hash mismatch")
        # The rollout engine may have been built from the initial weights;
        # refresh it after the resume load.
        trainer.engine.sync_weights(model.state_dict())
    return trainer, model


def _disable_indexer_aux(model: MiniDeepSeekModel) -> None:
    """RL/OPD do not supervise the CSA indexer.

    The indexer features are detached in ``Indexer.forward`` and the auxiliary
    KL is only consumed by the pretrain/SFT loss.  Leaving it enabled in an
    actor forward costs a chunked teacher pass and, with activation
    checkpointing, makes the recomputed graph differ from the forward graph
    (``CheckpointError: different metadata``).  Drop it for the whole RL model.
    """
    core = model._core()
    for layer in getattr(core, "layers", {}).values():
        attention = getattr(layer, "attention", None)
        if attention is None:
            continue
        attention.indexer_aux = None
        attention.use_indexer_aux = False


def _validate_rl_settings(config: RLConfig) -> None:
    """Reject options that the local GRPO trainer would otherwise ignore."""
    if config.rl.loss_type not in {"grpo", "cispo"}:
        raise ValueError(f"unknown GRPO loss_type {config.rl.loss_type!r}")
    if config.rl.kl_coeff != 0:
        raise ValueError("GRPO does not implement a KL reference penalty")
    if config.rl.min_response_tokens != 2:
        raise ValueError("GRPO does not implement min_response_tokens filtering")
    if config.train.grad_accum != 1:
        raise ValueError("GRPO does not implement gradient accumulation")
    if config.train.val_path:
        raise ValueError("GRPO does not implement a validation split")



def _merge_grpo_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-rank GRPO metrics into whole-job figures.

    Means are weighted by samples (reward/length/stop rates) or response tokens
    (loss/KL/ratio); std is combined from the per-rank means and variances.
    Only rank 0 logs, so without this a 2-GPU run reports one rank's half.
    """
    if len(records) <= 1:
        return records[0]
    merged = dict(records[0])
    samples = sum(float(record.get("samples", 0.0)) for record in records)
    tokens = sum(float(record.get("response_tokens", 0.0)) for record in records)
    merged["samples"] = samples
    merged["response_tokens"] = tokens

    def weighted(key: str, weight_key: str) -> float:
        denominator = sum(float(record.get(weight_key, 0.0)) for record in records)
        if denominator <= 0:
            return 0.0
        return (
            sum(
                float(record.get(key, 0.0)) * float(record.get(weight_key, 0.0))
                for record in records
            )
            / denominator
        )

    for key in (
        "reward_mean",
        "advantage_abs_mean",
        "finished_rate",
        "max_new_hit_rate",
        "response_length_mean",
    ):
        merged[key] = weighted(key, "samples")
    for key in ("clip_fraction", "ratio_mean", "approx_kl", "loss"):
        merged[key] = weighted(key, "response_tokens")
    if samples > 0:
        mean = weighted("reward_mean", "samples")
        second = sum(
            float(record.get("samples", 0.0))
            * (
                float(record.get("reward_std", 0.0)) ** 2
                + float(record.get("reward_mean", 0.0)) ** 2
            )
            for record in records
        ) / samples
        merged["reward_std"] = math.sqrt(max(second - mean * mean, 0.0))
    for key in {key for record in records for key in record}:
        if key in {"reward_mean", "reward_std", "response_tokens", "samples", "loss", "grad_norm", "lr"}:
            continue
        if key.startswith("reward_"):
            merged[key] = weighted(key, "samples")
        elif key.startswith("routing_"):
            merged[key] = weighted(key, "response_tokens")
    merged["grad_norm"] = max(float(record.get("grad_norm", 0.0)) for record in records)
    merged["weight_sync_logprob_diff"] = max(
        float(record.get("weight_sync_logprob_diff", 0.0)) for record in records
    )
    merged["rollout_weight_version"] = max(
        int(record.get("rollout_weight_version", 0)) for record in records
    )
    merged["lr"] = records[0].get("lr", 0.0)
    return merged


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
