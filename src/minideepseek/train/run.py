"""Pretraining / SFT runner driven by YAML configs.

Usage::

    python -m minideepseek.train.run --config configs/pretrain/debug_smoke.yaml
    python -m minideepseek.train.run --config configs/sft/math.yaml --mode sft
"""

from __future__ import annotations

import argparse
import json

from ..config import TrainConfig, load_config
from ..data.pretrain import PackedPretrainDataset
from ..data.sft import SFTDataset
from ..tokenizer import load_tokenizer
from .distributed import init_distributed
from .trainer import build_trainer


def _load_single_dataset(mode: str, path: str):
    return SFTDataset(path) if mode == "sft" else PackedPretrainDataset(path)


def _batch_factory(
    config: TrainConfig,
    mode: str,
    dataset,
    *,
    shard_rank: int,
    shard_world_size: int,
):
    total_steps = max(config.train.steps * max(config.train.grad_accum, 1), 1)

    def factory():
        if mode == "sft":
            from ..data.sft import iter_batches

            return iter_batches(
                dataset,
                batch_tokens=config.data.batch_tokens,
                batch_tokens_final=config.data.batch_tokens_final,
                batch_warmup_ratio=config.data.batch_warmup_ratio,
                total_steps=total_steps,
                shuffle=config.data.shuffle,
                seed=config.data.seed,
                shard_rank=shard_rank,
                shard_world_size=shard_world_size,
            )
        from ..data.pretrain import iter_batches

        return iter_batches(
            dataset,
            batch_tokens=config.data.batch_tokens,
            batch_tokens_final=config.data.batch_tokens_final,
            batch_warmup_ratio=config.data.batch_warmup_ratio,
            total_steps=total_steps,
            seq_len=config.model.seq_len,
            shuffle=config.data.shuffle,
            seed=config.data.seed,
            shard_rank=shard_rank,
            shard_world_size=shard_world_size,
        )

    return factory


def load_dataset(
    config: TrainConfig,
    mode: str,
    *,
    shard_rank: int = 0,
    shard_world_size: int = 1,
    path: str | None = None,
):
    explicit_path = path
    path = path or config.data.tokenized_path
    if config.data.sources and explicit_path is None:
        from ..data.mix import BatchSource, WeightedBatchMixer

        total_batches = max(config.train.steps * max(config.train.grad_accum, 1), 1)
        sources = []
        meta_sources = []
        for spec in config.data.sources:
            dataset = _load_single_dataset(mode, spec.path)
            sources.append(
                BatchSource(
                    name=spec.path,
                    factory=_batch_factory(
                        config,
                        mode,
                        dataset,
                        shard_rank=shard_rank,
                        shard_world_size=shard_world_size,
                    ),
                    weight=spec.weight,
                )
            )
            meta_sources.append(
                {
                    "path": spec.path,
                    "weight": spec.weight,
                    "domain": spec.domain,
                    "tokens": getattr(dataset, "num_tokens", None),
                }
            )
        mixer = WeightedBatchMixer(
            sources,
            total_batches=total_batches,
            seed=config.data.seed,
            shuffle=config.data.shuffle,
        )
        return {"meta": {"sources": meta_sources}}, mixer
    if not path:
        raise SystemExit(
            "data.tokenized_path or data.sources is required; "
            "run `python -m minideepseek.data.prepare` first"
        )
    dataset = _load_single_dataset(mode, path)
    batches = _batch_factory(
        config,
        mode,
        dataset,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
    )()
    return dataset, batches


def run(config: TrainConfig, mode: str = "pretrain") -> dict:
    tokenizer = load_tokenizer(vocab_limit=config.model.vocab_limit)
    context = init_distributed(config.distributed, config.model.device)
    dataset, batches = load_dataset(
        config,
        mode,
        shard_rank=context.rank,
        shard_world_size=context.world_size,
    )
    val_batches = None
    if config.train.val_path:
        _, val_batches = load_dataset(
            config,
            mode,
            path=config.train.val_path,
            shard_rank=0,
            shard_world_size=1,
        )
    trainer, model = build_trainer(
        config,
        batches,
        val_batches=val_batches,
        tokenizer_hash=tokenizer.sha256,
        distributed=context,
    )
    summary = trainer.train()
    checkpoint = trainer.save("final")
    report = {
        "mode": mode,
        "dataset": dataset.get("meta", {}) if isinstance(dataset, dict) else getattr(dataset, "meta", {}),
        "params": model.parameter_report(),
        "summary": summary,
        "checkpoint": str(checkpoint),
        "device": str(context.device),
        "distributed": {
            "enabled": context.enabled,
            "mode": config.distributed.mode,
            "rank": context.rank,
            "world_size": context.world_size,
        },
    }
    context.shutdown()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    args = parser.parse_args(argv)
    config = load_config(TrainConfig, args.config)
    report = run(config, args.mode)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
