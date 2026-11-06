"""GRPO / OPD command-line runners."""

from __future__ import annotations

import argparse
import json

from ..config import RLConfig, load_config
from ..data.grpo import load_records
from ..data.opd import load_opd_prompts
from ..tokenizer import load_tokenizer
from ..train.distributed import init_distributed
from .grpo import build_grpo_trainer
from .opd import build_opd_trainer
from .reward import make_reward_fn


def run_grpo(config: RLConfig) -> dict:
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
        summary = trainer.train()
        checkpoint = trainer.save("final")
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
    if not config.data.path:
        raise SystemExit("data.path is required for OPD")
    context = init_distributed(config.distributed, config.model.device)
    try:
        tokenizer = load_tokenizer(vocab_limit=config.model.vocab_limit)
        prompts = load_opd_prompts(config.data.path)
        trainer, student = build_opd_trainer(
            config, prompts, tokenizer=tokenizer, distributed=context
        )
        summary = trainer.train()
        checkpoint = trainer.save("final")
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
