"""Multi-teacher On-Policy Distillation (OPD).

The student samples its own trajectories, each teacher scores the *same*
student trajectory, and the student minimises a full-vocabulary reverse KL:

    L = sum_i w_i * E_t [ KL( pi_student(.|s_t) || pi_teacher_i(.|s_t) ) ]

The 6,400-token vocabulary makes the full-vocabulary KL cheap, which is why the
roadmap rejects verl's sampled-token k1/k3 estimators for this stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import RLConfig, TeacherSettings, resolve_device, set_seed
from ..data.grpo import GRPORecord
from ..data.opd import OPDPrompt
from ..model.checkpoint import load_checkpoint, save_checkpoint
from ..model.wrapper import MiniDeepSeekModel, build_model
from ..tokenizer import MiniDeepSeekTokenizer, load_tokenizer
from ..train.metrics import MetricsLogger
from ..train.optimizer import build_optimizer
from ..train.scheduler import WarmupDecaySchedule
from .rollout import RolloutSample, TorchRolloutEngine


@dataclass
class OPDStepStats:
    loss: float = 0.0
    kl_total: float = 0.0
    kl_by_teacher: dict[str, float] = field(default_factory=dict)
    response_tokens: int = 0
    response_length_mean: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    rollout_weight_version: int = 0
    weight_sync_logprob_diff: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        record = {
            "loss": self.loss,
            "kl_total": self.kl_total,
            "response_tokens": float(self.response_tokens),
            "response_length_mean": self.response_length_mean,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "rollout_weight_version": self.rollout_weight_version,
            "weight_sync_logprob_diff": self.weight_sync_logprob_diff,
        }
        for key, value in self.kl_by_teacher.items():
            record[f"kl_{key}"] = value
        return record


class OPDTrainer:
    def __init__(
        self,
        config: RLConfig,
        student: MiniDeepSeekModel,
        tokenizer: MiniDeepSeekTokenizer,
        prompts: Sequence[OPDPrompt],
        teachers: dict[str, MiniDeepSeekModel],
        *,
        teacher_weights: dict[str, float] | None = None,
        engine: TorchRolloutEngine | None = None,
    ):
        if not prompts:
            raise ValueError("OPD requires at least one prompt")
        if not teachers:
            raise ValueError("OPD requires at least one teacher")
        self.config = config
        self.student = student
        self.tokenizer = tokenizer
        self.prompts = list(prompts)
        self.teachers = teachers
        self.engine = engine or TorchRolloutEngine(student, tokenizer)
        self.optimizer = build_optimizer(student, config.optim)
        self.scheduler = WarmupDecaySchedule(
            self.optimizer,
            warmup_steps=config.schedule.warmup_steps,
            total_steps=max(config.train.steps, 1),
            decay=config.schedule.decay,
            min_lr_ratio=config.schedule.min_lr_ratio,
        )
        configured = dict(config.rl.teacher_weights)
        for teacher in config.teachers:
            configured.setdefault(teacher.key or teacher.name, teacher.weight)
        self.teacher_weights = teacher_weights or configured or {
            key: 1.0 for key in teachers
        }
        missing = set(teachers) - set(self.teacher_weights)
        if missing:
            raise ValueError(f"teachers without weights: {sorted(missing)}")
        self.logger = MetricsLogger(config.train.out_dir)
        self.step_index = 0
        set_seed(config.model.seed)

    def _sample_prompts(self) -> list[OPDPrompt]:
        import random

        rng = random.Random(self.config.model.seed + self.step_index)
        count = min(max(int(self.config.rl.prompts_per_step), 1), len(self.prompts))
        indices = list(range(len(self.prompts)))
        rng.shuffle(indices)
        return [self.prompts[index] for index in indices[:count]]

    def _rollout(self) -> list[RolloutSample]:
        prompts = self._sample_prompts()
        records = [
            GRPORecord(prompt=prompt.prompt, domain=prompt.domain, verifier="none")
            for prompt in prompts
        ]
        groups = self.engine.rollout(
            records,
            group_size=1,
            max_new_tokens=self.config.rl.max_new_tokens,
            temperature=self.config.rl.temperature,
            seed=self.config.model.seed + self.step_index,
        )
        samples: list[RolloutSample] = []
        for prompt, group in zip(prompts, groups, strict=True):
            for sample in group.samples:
                sample.teacher_key = prompt.teacher_key
                samples.append(sample)
        return samples

    def _kl_for_group(
        self,
        samples: Sequence[RolloutSample],
        teacher_key: str,
    ) -> tuple[Tensor, float, int]:
        """Reverse KL for one group that shares a single teacher key."""
        batch, response_mask = self.engine.build_batch(samples, self.student.device)
        positions = response_mask.nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            return torch.zeros((), device=self.student.device), 0.0, 0
        if int(positions.min()) == 0:
            raise RuntimeError("response token at position 0 has no prediction logit")
        prediction_positions = positions - 1
        student_logits = self.student.full_logits(
            batch, temperature=self.config.rl.temperature
        )[prediction_positions]
        student_logp = F.log_softmax(student_logits.float(), dim=-1)
        student_prob = student_logp.exp()
        teacher = self.teachers[teacher_key]
        with torch.no_grad():
            teacher.eval()
            teacher_logits = teacher.full_logits(
                batch, temperature=self.config.rl.temperature
            )[prediction_positions]
            teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
        kl = (student_prob * (student_logp - teacher_logp)).sum(dim=-1).mean()
        return kl, float(kl.detach()), int(positions.numel())

    def step(self) -> dict[str, Any]:
        samples = self._rollout()
        sync_diff = 0.0
        every = int(self.config.rl.weight_sync_check_every)
        if every > 0 and self.engine.generator is not None and self.step_index % every == 0:
            diff = self.engine.check_generator_consistency(samples)
            sync_diff = float(diff or 0.0)
            if sync_diff > self.config.rl.weight_sync_threshold:
                raise RuntimeError(
                    f"rollout/student weight-sync logprob mismatch: {sync_diff:.3e} "
                    f"> {self.config.rl.weight_sync_threshold:.1e}"
                )
        if not samples:
            raise RuntimeError("rollout produced no samples")
        self.optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=self.student.device)
        weight_sum = 0.0
        kl_by_teacher: dict[str, float] = {}
        token_count = 0
        for key, group in _group_by_teacher(samples).items():
            if key not in self.teachers:
                raise KeyError(
                    f"no teacher for key {key!r}; available={sorted(self.teachers)}"
                )
            kl, kl_value, tokens = self._kl_for_group(group, key)
            weight = float(self.teacher_weights.get(key, 1.0))
            loss_total = loss_total + kl * weight
            weight_sum += weight
            token_count += tokens
            kl_by_teacher[key] = kl_value
        if token_count == 0:
            raise RuntimeError("no response tokens in OPD batch")
        if weight_sum > 0:
            loss_total = loss_total / weight_sum
        if not torch.isfinite(loss_total):
            raise RuntimeError(f"non-finite OPD loss at step {self.step_index}")
        loss_total.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(self.student.parameters()), self.config.optim.grad_clip
            )
        )
        self.scheduler.step(self.step_index)
        self.optimizer.step()
        rollout_version = self.engine.sync_weights(self.student.state_dict())
        stats = OPDStepStats(
            loss=float(loss_total.detach()),
            kl_total=float(loss_total.detach()),
            kl_by_teacher=kl_by_teacher,
            response_tokens=token_count,
            response_length_mean=sum(sample.response_length for sample in samples)
            / len(samples),
            grad_norm=grad_norm,
            lr=float(self.optimizer.param_groups[0]["lr"]),
            rollout_weight_version=rollout_version,
            weight_sync_logprob_diff=sync_diff,
        )
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
            self.student,
            optimizer=self.optimizer,
            step=self.step_index,
            config=self.config.to_dict(),
            tokenizer_hash=self.tokenizer.sha256,
            seed=self.config.model.seed,
            extra={"method": "opd", "teachers": sorted(self.teachers)},
        )
        return path


def _group_by_teacher(samples: Sequence[RolloutSample]) -> dict[str, list[RolloutSample]]:
    grouped: dict[str, list[RolloutSample]] = {}
    for sample in samples:
        key = sample.teacher_key or "general"
        grouped.setdefault(key, []).append(sample)
    return grouped


def load_teacher(
    settings: TeacherSettings,
    *,
    tokenizer: MiniDeepSeekTokenizer | None = None,
    device: torch.device | str = "cpu",
) -> MiniDeepSeekModel:
    tokenizer = tokenizer or load_tokenizer()
    model = build_model(
        settings.flavor,
        seq_len=settings.seq_len,
        n_mtp_layers=settings.n_mtp_layers,
        device=device,
        swiglu_limit=10.0,
        dtype=settings.dtype,
    )
    if settings.checkpoint:
        meta = load_checkpoint(settings.checkpoint, model, map_location="cpu")
        expected = meta.get("tokenizer_hash")
        if expected and expected != tokenizer.sha256:
            raise ValueError(
                f"teacher {settings.name} tokenizer hash mismatch: "
                f"{expected} != {tokenizer.sha256}"
            )
    model.eval()
    return model


def build_opd_trainer(
    config: RLConfig,
    prompts: Sequence[OPDPrompt],
    *,
    tokenizer: MiniDeepSeekTokenizer | None = None,
    teacher_models: dict[str, MiniDeepSeekModel] | None = None,
) -> tuple[OPDTrainer, MiniDeepSeekModel]:
    tokenizer = tokenizer or load_tokenizer(vocab_limit=config.model.vocab_limit)
    device = resolve_device(config.model.device)
    student = build_model(
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
    if config.train.resume:
        meta = load_checkpoint(config.train.resume, student, map_location=device)
        expected = meta.get("tokenizer_hash")
        if expected and expected != tokenizer.sha256:
            raise ValueError("OPD student checkpoint tokenizer hash mismatch")
    teachers = teacher_models or {}
    if not teachers:
        for settings in config.teachers:
            teachers[settings.key or settings.name] = load_teacher(
                settings, tokenizer=tokenizer, device=device
            )
    from .backends import build_rollout_engine

    engine = build_rollout_engine(config, student, tokenizer)
    trainer = OPDTrainer(
        config, student, tokenizer, prompts, teachers, engine=engine
    )
    return trainer, student
