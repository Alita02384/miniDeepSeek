"""GRPO / OPD command-line runners."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

# RL batches have many distinct packed lengths; the fused attn_gym kernel
# specializes on every layout and would recompile mid-step (OPD teacher
# forwards hung for minutes on the first step).  The torch reference is exact
# and has no compile cliff, so make it the RL default.  SFT/pretrain keep the
# fused path via their own configs.
os.environ.setdefault("MINIDEEPSEEK_ATTN_BACKEND", "torch")

from ..config import RLConfig, load_config
from ..data.grpo import load_records
from ..data.opd import load_opd_prompts
from ..tokenizer import load_tokenizer
from ..train.distributed import init_distributed
from ..train.trainer import latest_checkpoint
from .grpo import build_grpo_trainer
from .opd import build_opd_trainer
from .reward import make_reward_fn


def _resume_path(config: RLConfig) -> str | None:
    """Prefer the newest checkpoint in out_dir when auto_resume is on.

    RL configs also set ``resume`` to the SFT/pretrain base, so the old
    ``if resume is None`` test made auto_resume dead code.  A checkpoint in
    out_dir is a crash-recovery checkpoint: resume it and restore its step.
    """
    resume = config.train.resume
    if config.train.auto_resume:
        found = latest_checkpoint(config.train.out_dir)
        if found is not None:
            config.train.resume_step = True
            return str(found)
    return resume


def _prune_checkpoints(out_dir: str, keep: int) -> None:
    if keep <= 0:
        return
    directory = Path(out_dir)
    if not directory.is_dir():
        return
    periodic = sorted(
        (
            child
            for child in directory.iterdir()
            if child.is_dir() and re.fullmatch(r"step_\d+", child.name)
        ),
        key=lambda path: int(path.name.split("_")[1]),
    )
    for old in periodic[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def run_grpo(config: RLConfig) -> dict:
    if config.rl.method != "grpo":
        raise SystemExit(f"config.rl.method is {config.rl.method!r}, expected 'grpo'")
    if not config.data.path:
        raise SystemExit("data.path is required for GRPO")
    context = init_distributed(config.distributed, config.model.device)
    try:
        tokenizer = load_tokenizer(vocab_limit=config.model.vocab_limit)
        records = load_records(config.data.path)
        reward_fn = make_reward_fn(config.reward)
        trainer, model = build_grpo_trainer(
            config, records, reward_fn, tokenizer=tokenizer, distributed=context
        )
        resume = _resume_path(config)
        if resume:
            trainer.resume(resume)
        summary = trainer.train()
        checkpoint = Path(config.train.out_dir) / "final"
        _prune_checkpoints(config.train.out_dir, int(config.train.keep_last_ckpt))
        return {
            "method": "grpo",
            "records": len(records),
            "params": model.parameter_report(),
            "summary": summary,
            "checkpoint": str(checkpoint),
            "distributed": {
                "enabled": context.enabled,
                "rank": context.rank,
                "world_size": context.world_size,
            },
        }
    finally:
        context.shutdown()


def run_opd(config: RLConfig) -> dict:
    if config.rl.method != "opd":
        raise SystemExit(f"config.rl.method is {config.rl.method!r}, expected 'opd'")
    if not config.data.path:
        raise SystemExit("data.path is required for OPD")
    context = init_distributed(config.distributed, config.model.device)
    try:
        tokenizer = load_tokenizer(vocab_limit=config.model.vocab_limit)
        prompts = load_opd_prompts(config.data.path)
        trainer, student = build_opd_trainer(
            config, prompts, tokenizer=tokenizer, distributed=context
        )
        resume = _resume_path(config)
        if resume:
            trainer.resume(resume)
        summary = trainer.train()
        checkpoint = Path(config.train.out_dir) / "final"
        _prune_checkpoints(config.train.out_dir, int(config.train.keep_last_ckpt))
        return {
            "method": "opd",
            "prompts": len(prompts),
            "teachers": sorted(trainer.teachers),
            "params": student.parameter_report(),
            "summary": summary,
            "checkpoint": str(checkpoint),
            "distributed": {
                "enabled": context.enabled,
                "rank": context.rank,
                "world_size": context.world_size,
            },
        }
    finally:
        context.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["grpo", "opd"])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model-config",
        default=None,
        help="model YAML overriding the config's model_file",
    )
    args = parser.parse_args(argv)
    config = load_config(RLConfig, args.config, model_file=args.model_config)
    report = run_grpo(config) if args.method == "grpo" else run_opd(config)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
