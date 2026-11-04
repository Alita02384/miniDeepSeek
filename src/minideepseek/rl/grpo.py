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
from ..train.metrics import MetricsLogger
from ..train.optimizer import build_optimizer
from ..train.scheduler import WarmupDecaySchedule
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
    ):
        if not records:
            raise ValueError("GRPO requires at least one prompt record")
        if config.rl.group_size < 2:
            raise ValueError("GRPO needs group_size >= 2 to compute advantages")
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.records = list(records)
        self.reward_fn = reward_fn
        self.engine = engine or TorchRolloutEngine(model, tokenizer)
        self.optimizer = build_optimizer(model, config.optim)
        self.scheduler = WarmupDecaySchedule(
            self.optimizer,
            warmup_steps=config.schedule.warmup_steps,
            total_steps=max(config.train.steps, 1),
            decay=config.schedule.decay,
            min_lr_ratio=config.schedule.min_lr_ratio,
        )
        self.logger = MetricsLogger(config.train.out_dir)
        self.rng = random.Random(config.model.seed)
        self.step_index = 0
        self.run_name = run_name or Path(config.train.out_dir).name
        set_seed(config.model.seed)

    def _sample_records(self) -> list[GRPORecord]:
        count = min(max(int(self.config.rl.prompts_per_step), 1), len(self.records))
        indices = list(range(len(self.records)))
        self.rng.shuffle(indices)
        return [self.records[index] for index in indices[:count]]

    def _flatten(self, groups: list[RolloutGroup]) -> list[RolloutSample]:
        return [sample for group in groups for sample in group.samples]

    def compute_loss(
        self,
        groups: list[RolloutGroup],
        rewards: list[list[RewardResult]],
    ) -> tuple[Tensor, GRPOStepStats]:
        flat_samples = self._flatten(groups)
        flat_rewards = [result.total for group in rewards for result in group]
        advantages: list[float] = []
        for group_rewards in rewards:
            advantages.extend(group_advantages([result.total for result in group_rewards]))
        if len(advantages) != len(flat_samples):
            raise RuntimeError("advantage/sample accounting mismatch")

        batch, response_mask = self.engine.build_batch(flat_samples, self.model.device)
        length = sum(sample.response_length for sample in flat_samples)
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
        old_logprobs = torch.tensor(
            [value for sample in flat_samples for value in sample.sampling_logprobs],
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
                for _ in range(sample.response_length)
            ],
            dtype=torch.float32,
            device=self.model.device,
        )
        ratio = torch.exp(actor_logprobs - old_logprobs)
        clip = float(self.config.rl.clip)
        unclipped = ratio * advantage_tensor
        clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage_tensor
        policy_loss = -torch.minimum(unclipped, clipped).sum() / max(length, 1)
        with torch.no_grad():
            clip_fraction = (
                ((ratio < 1.0 - clip) | (ratio > 1.0 + clip)).float().mean()
            )
            approx_kl = (old_logprobs - actor_logprobs).mean()
        stats = GRPOStepStats(
            loss=float(policy_loss.detach()),
            reward_mean=sum(flat_rewards) / len(flat_rewards),
            reward_std=_std(flat_rewards),
            advantage_abs_mean=sum(abs(value) for value in advantages) / len(advantages),
            response_tokens=int(length),
            response_length_mean=sum(sample.response_length for sample in flat_samples)
            / len(flat_samples),
            clip_fraction=float(clip_fraction),
            ratio_mean=float(ratio.detach().mean()),
            approx_kl=float(approx_kl),
            routing=routing_stats,
        )
        component_values: dict[str, list[float]] = {}
        for group in rewards:
            for result in group:
                for key, value in result.components.items():
                    component_values.setdefault(key, []).append(float(value))
        stats.components = {
            key: sum(values) / len(values) for key, values in component_values.items()
        }
        return policy_loss, stats

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
            [self.reward_fn(sample.response_text, group.record) for sample in group.samples]
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
        loss, stats = self.compute_loss(groups, rewards)
        stats.weight_sync_logprob_diff = sync_diff
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite GRPO loss at step {self.step_index}")
        loss.backward()
        stats.grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()), self.config.optim.grad_clip
            )
        )
        self.scheduler.step(self.step_index)
        self.optimizer.step()
        stats.lr = float(self.optimizer.param_groups[0]["lr"])
        return stats.as_dict()

    def train(self, steps: int | None = None) -> dict[str, Any]:
        total = int(steps if steps is not None else self.config.train.steps)
        while self.step_index < total:
            metrics = self.step()
            if self.step_index % max(self.config.train.log_every, 1) == 0:
                self.logger.log(self.step_index, metrics)
            if (
                self.config.train.ckpt_every > 0
                and (self.step_index + 1) % self.config.train.ckpt_every == 0
            ):
                self.save(f"step_{self.step_index + 1:06d}")
            self.step_index += 1
        self.save("final")
        summary = self.logger.summary()
        summary["steps_run"] = self.step_index
        return summary

    def save(self, name: str = "final") -> Path:
        path = Path(self.config.train.out_dir) / name
        save_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            step=self.step_index,
            config=self.config.to_dict(),
            tokenizer_hash=self.tokenizer.sha256,
            seed=self.config.model.seed,
            extra={"run_name": self.run_name, "method": "grpo"},
        )
        return path


def build_grpo_trainer(
    config: RLConfig,
    records: Sequence[GRPORecord],
    reward_fn: Callable[[str, GRPORecord], RewardResult],
    *,
    tokenizer: MiniDeepSeekTokenizer | None = None,
) -> tuple[GRPOTrainer, MiniDeepSeekModel]:
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
        dtype=config.model.dtype,
    )
    tokenizer = tokenizer or load_tokenizer(vocab_limit=config.model.vocab_limit)
    if config.train.resume:
        meta = load_checkpoint(config.train.resume, model, map_location=device)
        expected = meta.get("tokenizer_hash")
        if expected and expected != tokenizer.sha256:
            raise ValueError("GRPO resume checkpoint tokenizer hash mismatch")
    from .backends import build_rollout_engine

    engine = build_rollout_engine(config, model, tokenizer)
    trainer = GRPOTrainer(
        config, model, tokenizer, records, reward_fn, engine=engine
    )
    return trainer, model


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
