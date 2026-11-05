"""Sequence-length curriculum runner.

V4 extends pre-training from 4K to 16K/64K/1M in stages.  This project keeps
the same idea at 2K/4K/8K scale: each stage has its own packed dataset (the
tokenizer/packing length is chosen by the data pipeline), a cumulative step
target, and resumes the previous stage's checkpoint.

The runner is deliberately a thin orchestration layer over ``train.run``; it
does not mutate a live model's RoPE cache mid-run, which avoids the subtle
position/bookkeeping bugs of dynamically changing packing length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import BaseConfig, TrainConfig, load_config

__all__ = [
    "CurriculumSettings",
    "CurriculumStage",
    "run_curriculum",
    "validate_stages",
]


@dataclass
class CurriculumStage(BaseConfig):
    """One sequence-length stage.

    ``until_step`` is cumulative: stage 2 continues until that optimizer step
    (the previous stage must end at a smaller value).
    """

    seq_len: int
    tokenized_path: str
    until_step: int
    batch_tokens: int | None = None
    batch_tokens_final: int | None = None
    batch_warmup_ratio: float | None = None
    index_topk: int | None = None


@dataclass
class CurriculumSettings(BaseConfig):
    base_config: str
    stages: list[CurriculumStage] = field(default_factory=list)
    out_dir: str = "checkpoints/curriculum"
    mode: str = "pretrain"


def validate_stages(stages: list[CurriculumStage], *, max_context: int = 8192) -> None:
    if not stages:
        raise ValueError("curriculum needs at least one stage")
    previous_step = 0
    previous_len = 0
    for index, stage in enumerate(stages):
        if stage.seq_len <= 0 or stage.seq_len > max_context:
            raise ValueError(
                f"stage {index}: seq_len must be in [1, {max_context}], got {stage.seq_len}"
            )
        if stage.until_step <= previous_step:
            raise ValueError(
                f"stage {index}: until_step must increase; "
                f"{stage.until_step} <= {previous_step}"
            )
        if stage.seq_len < previous_len:
            raise ValueError(
                f"stage {index}: sequence length must not shrink in a curriculum"
            )
        if not stage.tokenized_path:
            raise ValueError(f"stage {index}: tokenized_path is required")
        if stage.batch_tokens is not None and stage.batch_tokens <= 0:
            raise ValueError(f"stage {index}: batch_tokens must be positive")
        if stage.batch_warmup_ratio is not None and not 0 <= stage.batch_warmup_ratio <= 1:
            raise ValueError(f"stage {index}: batch_warmup_ratio must be in [0, 1]")
        if stage.index_topk is not None and stage.index_topk <= 0:
            raise ValueError(f"stage {index}: index_topk must be positive")
        previous_step = stage.until_step
        previous_len = stage.seq_len


def _stage_config(
    base: TrainConfig,
    stage: CurriculumStage,
    *,
    out_dir: Path,
    index: int,
    resume: str | None,
    schedule_steps: int,
) -> TrainConfig:
    config = TrainConfig.from_dict(base.to_dict())
    config.model.seq_len = int(stage.seq_len)
    if stage.index_topk is not None:
        config.model.index_topk = int(stage.index_topk)
    config.data.tokenized_path = stage.tokenized_path
    if stage.batch_tokens is not None:
        config.data.batch_tokens = int(stage.batch_tokens)
    if stage.batch_tokens_final is not None:
        config.data.batch_tokens_final = int(stage.batch_tokens_final)
    elif config.data.batch_tokens_final is not None:
        # Keep the final budget at least as large as the stage base.
        config.data.batch_tokens_final = max(
            int(config.data.batch_tokens_final), int(config.data.batch_tokens)
        )
    if stage.batch_warmup_ratio is not None:
        config.data.batch_warmup_ratio = float(stage.batch_warmup_ratio)
    config.train.steps = int(stage.until_step)
    config.train.schedule_steps = int(schedule_steps)
    config.train.out_dir = str(out_dir / f"stage_{index:02d}_seq{stage.seq_len}")
    config.train.resume = resume
    # Each stage has its own packed dataset/order.
    config.train.resume_data = False
    return config


def run_curriculum(settings: CurriculumSettings) -> list[dict[str, Any]]:
    """Run all stages in order, resuming each from the previous checkpoint."""
    validate_stages(settings.stages)
    base = load_config(TrainConfig, settings.base_config)
    out_dir = Path(settings.out_dir)
    reports: list[dict[str, Any]] = []
    resume: str | None = None
    final_step = int(settings.stages[-1].until_step)
    for index, stage in enumerate(settings.stages):
        config = _stage_config(
            base,
            stage,
            out_dir=out_dir,
            index=index,
            resume=resume,
            schedule_steps=final_step,
        )
        from .run import run as run_stage

        report = run_stage(config, settings.mode)
        reports.append(
            {
                "stage": index,
                "seq_len": stage.seq_len,
                "until_step": stage.until_step,
                "checkpoint": report.get("checkpoint"),
                "last_loss": report.get("summary", {}).get("last_loss"),
            }
        )
        resume = report.get("checkpoint")
    return reports
