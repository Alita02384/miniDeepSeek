"""Multi-teacher On-Policy Distillation (OPD).

The student samples its own trajectories, each teacher scores the *same*
student trajectory, and the student minimises a full-vocabulary reverse KL:

    L = sum_i w_i * E_t [ KL( pi_student(.|s_t) || pi_teacher_i(.|s_t) ) ]

The 6,400-token vocabulary makes the full-vocabulary KL cheap, which is why the
roadmap rejects verl's sampled-token k1/k3 estimators for this stage.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import (
    ModelSettings,
    RLConfig,
    TeacherSettings,
    load_config,
    resolve_device,
    set_seed,
)
from ..data.grpo import GRPORecord
from ..data.opd import OPDPrompt
from ..model.checkpoint import load_checkpoint, read_metadata, save_checkpoint
from ..model.config import load_model_settings
from ..model.wrapper import MiniDeepSeekModel, build_model
from ..tokenizer import MiniDeepSeekTokenizer, load_tokenizer
from ..train.distributed import (
    DistributedContext,
    all_reduce_gradients,
    apply_parallelism,
    clip_grad_norm,
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
        distributed: DistributedContext | None = None,
    ):
        if not prompts:
            raise ValueError("OPD requires at least one prompt")
        if not teachers:
            raise ValueError("OPD requires at least one teacher")
        _validate_rl_settings(config)
        self.config = config
        self.student = student
        self.tokenizer = tokenizer
        self.dist = distributed or DistributedContext(device=student.device)
        self.prompts = list(prompts)
        self.teachers = teachers
        if engine is None:
            if any(prompt.multi_turn or prompt.domain == "tool" for prompt in prompts):
                engine = ToolRolloutEngine(
                    student,
                    tokenizer,
                    max_turns=config.rl.max_turns,
                    thinking_ratio=config.rl.thinking_ratio,
                    max_tool_calls=config.rl.max_tool_calls,
                )
            else:
                engine = TorchRolloutEngine(student, tokenizer)
        self.engine = engine
        self.optimizer = build_optimizer(student, config.optim)
        self.scheduler = WarmupDecaySchedule(
            self.optimizer,
            warmup_steps=config.schedule.warmup_steps,
            total_steps=max(config.train.schedule_steps or config.train.steps, 1),
            decay=config.schedule.decay,
            min_lr_ratio=config.schedule.min_lr_ratio,
            decay_start_ratio=config.schedule.decay_start_ratio,
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
        self.logger = (
            MetricsLogger(config.train.out_dir) if self.dist.is_main else None
        )
        self.step_index = 0
        set_seed(config.model.seed)

    def _sample_prompts(self) -> list[OPDPrompt]:
        import random

        rng = random.Random(self.config.model.seed + self.step_index)
        # Global prompt budget split across DP ranks; unchanged at world=1.
        world = max(int(self.dist.world_size), 1)
        count = min(
            max(int(self.config.rl.prompts_per_step) // world, 1), len(self.prompts)
        )
        indices = list(range(len(self.prompts)))
        rng.shuffle(indices)
        return [self.prompts[index] for index in indices[:count]]

    def _rollout(self) -> list[RolloutSample]:
        prompts = self._sample_prompts()
        records = [
            GRPORecord(
                prompt=prompt.prompt,
                domain=prompt.domain,
                verifier="none",
                multi_turn=prompt.multi_turn,
            )
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

    def _update_expert_biases(self) -> None:
        # RL also increments the router usage counters in the actor/student
        # forward; settle them before the optimizer step just like the
        # pretrain/SFT trainer does.
        if self.dist.enabled:
            for _name, moe in iter_moe_layers(self.student.model):
                self.dist.all_reduce_sum(moe.router.tokens_per_expert_E)
        update_expert_biases(self.student.model)

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
        if self.dist.enabled and self.dist.mode == "ddp":
            all_reduce_gradients(self.student.model, self.dist.world_size)
        grad_norm = clip_grad_norm(self.student.model, self.config.optim.grad_clip)
        self._update_expert_biases()
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
            state = gather_model_state_dict(self.student.model)
            optim_state = gather_optimizer_state_dict(self.student.model, self.optimizer)
            self.dist.barrier()
            if self.dist.is_main:
                save_checkpoint(
                    path,
                    self.student,
                    optimizer=self.optimizer,
                    step=save_step,
                    config=self.config.to_dict(),
                    tokenizer_hash=self.tokenizer.sha256,
                    seed=self.config.model.seed,
                    extra={
                        "method": "opd",
                        "teachers": sorted(self.teachers),
                        "index_topk": self.student.current_index_topk(),
                    },
                    state_dict=state,
                    optimizer_state=optim_state,
                )
            self.dist.barrier()
        else:
            save_checkpoint(
                path,
                self.student,
                optimizer=self.optimizer,
                step=save_step,
                config=self.config.to_dict(),
                tokenizer_hash=self.tokenizer.sha256,
                seed=self.config.model.seed,
                extra={
                    "method": "opd",
                    "teachers": sorted(self.teachers),
                    "index_topk": self.student.current_index_topk(),
                },
            )
        return path

    def resume(self, directory: str | Path) -> dict[str, Any]:
        # D22: load model weights only; see GRPOTrainer.resume.
        if not self.dist.enabled:
            meta = load_checkpoint(
                directory,
                self.student,
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
            load_model_state_dict(self.student.model, state)
            meta = read_metadata(path)
        self.step_index = (
            int(meta.get("step", 0)) if self.config.train.resume_step else 0
        )
        return meta


def _validate_rl_settings(config: RLConfig) -> None:
    """Reject options that the local OPD trainer would otherwise ignore."""
    if config.rl.kl_coeff != 0:
        raise ValueError("OPD does not implement an extra KL reference penalty")
    if config.rl.min_response_tokens != 2:
        raise ValueError("OPD does not implement min_response_tokens filtering")
    if config.train.grad_accum != 1:
        raise ValueError("OPD does not implement gradient accumulation")
    if config.train.val_path:
        raise ValueError("OPD does not implement a validation split")


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
    index_topk = settings.index_topk
    if settings.checkpoint and index_topk is None:
        index_topk = read_metadata(settings.checkpoint).get("extra", {}).get("index_topk")
    if settings.model_file:
        model_settings = load_config(ModelSettings, settings.model_file)
    else:
        model_settings = load_model_settings(settings.flavor)
    updates: dict[str, Any] = {
        "seq_len": int(settings.seq_len),
        "n_mtp_layers": int(settings.n_mtp_layers),
    }
    if index_topk is not None:
        updates["index_topk"] = int(index_topk)
    model_settings = dataclasses.replace(model_settings, **updates)
    model = build_model(model_settings, device=device, dtype=settings.dtype)
    if settings.checkpoint:
        meta = load_checkpoint(
            settings.checkpoint,
            model,
            map_location="cpu",
            ignore_prefixes=("mtp_layers.",),
        )
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
    distributed: DistributedContext | None = None,
) -> tuple[OPDTrainer, MiniDeepSeekModel]:
    context = distributed or init_distributed(config.distributed, config.model.device)
    tokenizer = tokenizer or load_tokenizer(vocab_limit=config.model.vocab_limit)
    device = context.device if context.enabled else resolve_device(config.model.device)
    set_seed(config.model.seed)
    student = build_model(
        config.model,
        device=device,
        dtype=config.model.dtype,
    )
    if context.enabled:
        student.model = apply_parallelism(
            student.model, context, config.distributed, device=device
        )
        if config.rl.rollout_backend == "vllm":
            raise ValueError(
                "vLLM rollout with FSDP2 OPD is not wired; use the torch backend "
                "or run verl's FSDP actor."
            )
        prompts = list(prompts)[context.rank :: context.world_size]
        if not prompts:
            raise ValueError(
                f"rank {context.rank} has no OPD prompts after sharding"
            )
    teachers = teacher_models or {}
    if not teachers:
        for settings in config.teachers:
            teachers[settings.key or settings.name] = load_teacher(
                settings, tokenizer=tokenizer, device=device
            )
    from .backends import build_rollout_engine

    engine = build_rollout_engine(config, student, tokenizer)
    if any(prompt.multi_turn or prompt.domain == "tool" for prompt in prompts):
        engine = ToolRolloutEngine(
            student,
            tokenizer,
            generator=getattr(engine, "generator", None),
            max_turns=config.rl.max_turns,
            thinking_ratio=config.rl.thinking_ratio,
            max_tool_calls=config.rl.max_tool_calls,
        )
    trainer = OPDTrainer(
        config,
        student,
        tokenizer,
        prompts,
        teachers,
        engine=engine,
        distributed=context,
    )
    if config.train.resume:
        meta = trainer.resume(config.train.resume)
        expected = meta.get("tokenizer_hash")
        if expected and expected != tokenizer.sha256:
            raise ValueError("OPD student checkpoint tokenizer hash mismatch")
        # Refresh a sampler built before the resume load (torch engine is a no-op).
        trainer.engine.sync_weights(student.state_dict())
    return trainer, student
