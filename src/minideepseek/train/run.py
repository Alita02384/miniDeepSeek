"""Pretraining / SFT runner driven by YAML configs.

Usage::

    python -m minideepseek.train.run --config configs/pretrain/debug_smoke.yaml
    python -m minideepseek.train.run --config configs/sft/math.yaml --mode sft
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator

from ..config import TrainConfig, load_config
from ..data.pretrain import PackedPretrainDataset
from ..data.sft import SFTDataset
from ..model.batch import PackedBatch
from ..tokenizer import load_tokenizer
from .distributed import init_distributed
from .trainer import build_trainer


def _load_single_dataset(mode: str, path: str):
    return SFTDataset(path) if mode == "sft" else PackedPretrainDataset(path)


def _per_rank_tokens(value: int | None, world_size: int) -> int | None:
    """Split a configured global micro-batch token budget across DP ranks."""
    if value is None:
        return None
    return max(1, int(value) // max(int(world_size), 1))


def _batch_factory(
    config: TrainConfig,
    mode: str,
    dataset,
    *,
    shard_rank: int,
    shard_world_size: int,
) -> Callable[[], Iterator[PackedBatch]]:
    total_steps = max(config.train.steps * max(config.train.grad_accum, 1), 1)
    # ``batch_tokens`` is a global (all-rank) micro-batch budget in the
    # shipped configs; each rank gets its shard.  ``total_steps`` stays in
    # optimizer/micro-batch units, so only the per-micro token count changes.
    batch_tokens = _per_rank_tokens(config.data.batch_tokens, shard_world_size)
    batch_tokens_final = _per_rank_tokens(
        config.data.batch_tokens_final, shard_world_size
    )

    def factory() -> Iterator[PackedBatch]:
        if mode == "sft":
            from ..data.sft import iter_batches

            return iter_batches(
                dataset,
                batch_tokens=batch_tokens,
                batch_tokens_final=batch_tokens_final,
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
            batch_tokens=batch_tokens,
            batch_tokens_final=batch_tokens_final,
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
) -> tuple[object, Callable[[], Iterator[PackedBatch]]]:
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
        return {"meta": {"sources": meta_sources}}, (lambda: iter(mixer))
    if not path:
        raise SystemExit(
            "data.tokenized_path or data.sources is required; "
            "run `python -m minideepseek.data.prepare` first"
        )
    dataset = _load_single_dataset(mode, path)
    factory = _batch_factory(
        config,
        mode,
        dataset,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
    )
    return dataset, factory



def _validate_dataset(dataset, tokenizer, config: TrainConfig, *, label: str) -> None:
    """Fail early on tokenizer/sequence-length drift between data and model."""
    meta = getattr(dataset, "meta", None)
    if not isinstance(meta, dict):
        return
    expected_hash = meta.get("tokenizer_hash")
    if expected_hash and expected_hash != tokenizer.sha256:
        raise ValueError(
            f"{label} dataset tokenizer hash mismatch: "
            f"{expected_hash} != {tokenizer.sha256}"
        )
    expected_vocab = meta.get("tokenizer_vocab")
    if expected_vocab and int(expected_vocab) != tokenizer.vocab_size:
        raise ValueError(
            f"{label} dataset tokenizer vocab {expected_vocab} does not match "
            f"loaded tokenizer {tokenizer.vocab_size}"
        )
    dataset_seq_len = meta.get("seq_len")
    if dataset_seq_len and int(dataset_seq_len) > int(config.model.seq_len):
        raise ValueError(
            f"{label} dataset packs up to {dataset_seq_len} tokens but the model "
            f"context is {config.model.seq_len}"
        )


def run(config: TrainConfig, mode: str = "pretrain") -> dict:
    tokenizer = load_tokenizer(vocab_limit=config.model.vocab_limit)
    context = init_distributed(config.distributed, config.model.device)
    dataset, batches = load_dataset(
        config,
        mode,
        shard_rank=context.rank,
        shard_world_size=context.world_size,
    )
    _validate_dataset(dataset, tokenizer, config, label="train")
    val_batches = None
    if config.train.val_path:
        val_dataset, val_batches = load_dataset(
            config,
            mode,
            path=config.train.val_path,
            shard_rank=0,
            shard_world_size=1,
        )
        _validate_dataset(val_dataset, tokenizer, config, label="val")
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
