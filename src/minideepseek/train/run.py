"""Pretraining / SFT runner driven by YAML configs.

Usage::

    python -m minideepseek.train.run --config configs/pretrain/debug_smoke.yaml
    python -m minideepseek.train.run --config configs/sft/math.yaml --mode sft
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..config import TrainConfig, load_config
from ..data.pretrain import BatchTokenSchedule, PackedPretrainDataset
from ..data.sft import SFTDataset
from ..model.batch import PackedBatch
from ..tokenizer import load_tokenizer
from .distributed import init_distributed
from .trainer import build_trainer


def _load_single_dataset(mode: str, path: str, *, full_packs: bool = False):
    if mode == "sft":
        return SFTDataset(path)
    return PackedPretrainDataset(path, full_packs=full_packs)


@dataclass
class MixedDataset:
    """Per-source datasets behind a weighted ``data.sources`` mixture.

    ``Trainer`` only needs the mixed ``meta`` for reporting; keeping the child
    datasets lets ``run`` validate each source's tokenizer/sequence length.
    """

    meta: dict
    datasets: list = field(default_factory=list)


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
    persistent_warmup: bool = True,
) -> Callable[[], Iterator[PackedBatch]]:
    total_steps = max(config.train.steps * max(config.train.grad_accum, 1), 1)
    # ``batch_tokens`` is a global (all-rank) micro-batch budget in the
    # shipped configs; each rank gets its shard.  ``total_steps`` stays in
    # optimizer/micro-batch units, so only the per-micro token count changes.
    batch_tokens = _per_rank_tokens(config.data.batch_tokens, shard_world_size)
    batch_tokens_final = _per_rank_tokens(
        config.data.batch_tokens_final, shard_world_size
    )
    schedule = None
    if persistent_warmup and batch_tokens is not None:
        # Shared by every restart of this loader, so an epoch boundary or a
        # resume does not reset the batch-token warmup to ``base``.
        schedule = BatchTokenSchedule(
            base=batch_tokens,
            final=batch_tokens_final,
            warmup_ratio=config.data.batch_warmup_ratio,
            total_steps=total_steps,
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
                schedule=schedule,
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
            schedule=schedule,
        )

    return factory


def _mixer_factory(
    sources: list,
    *,
    total_batches: int,
    seed: int,
    shuffle: bool,
    batches_per_step: int,
    lockstep: bool,
):
    """Build a *restartable* iterator factory over the domain mixer.

    The mixer's schedule is finite (``total_batches`` draws).  Handing back
    ``iter(existing_mixer)`` makes a restart return an iterator whose cursor is
    already at the end -- and ``Trainer._pull_batch`` restarts the factory
    whenever the loader runs dry, which is exactly what happens when resume
    replay consumes the last slot of a stage.  That killed the run at the
    2K -> 4K boundary, so the factory rebuilds the mixer instead.
    """

    from ..data.mix import WeightedBatchMixer

    def factory():
        mixer = WeightedBatchMixer(
            sources,
            total_batches=total_batches,
            seed=seed,
            shuffle=shuffle,
            batches_per_step=batches_per_step,
            lockstep=lockstep,
        )
        return iter(mixer)

    return factory


def load_dataset(
    config: TrainConfig,
    mode: str,
    *,
    shard_rank: int = 0,
    shard_world_size: int = 1,
    path: str | None = None,
    persistent_warmup: bool = True,
) -> tuple[object, Callable[[], Iterator[PackedBatch]]]:
    explicit_path = path
    # Validation sets (an explicit path) keep the packer's document-aligned
    # grid so val_loss stays comparable across a full_packs switch.
    train_full_packs = bool(config.data.full_packs) and explicit_path is None
    path = path or config.data.tokenized_path
    if config.data.sources and explicit_path is None:
        from ..data.mix import BatchSource

        total_batches = max(config.train.steps * max(config.train.grad_accum, 1), 1)
        sources = []
        meta_sources = []
        child_datasets = []
        seen_names: dict[str, int] = {}
        for spec in config.data.sources:
            dataset = _load_single_dataset(mode, spec.path, full_packs=train_full_packs)
            child_datasets.append(dataset)
            # Log by domain: ``trainer`` records ``domain_id`` from the active
            # source so per-step routing metrics are already per-domain.
            base_name = spec.domain or Path(spec.path).stem
            seen_names[base_name] = seen_names.get(base_name, 0) + 1
            name = (
                base_name
                if seen_names[base_name] == 1
                else f"{base_name}#{seen_names[base_name]}"
            )
            # ``data.sources[].weight`` is documented as an *input-token*
            # weight, but SFT's response-only mask makes the target/input ratio
            # very different per domain (general ~0.86, code ~0.37, tool
            # ~0.18).  Weighting batches by input tokens therefore gives the
            # tool expert only 33% of the loss tokens even at nominal 70%.
            # ``WeightedBatchMixer`` allocates *batches*, so divide the nominal
            # weight by the source's loss-token ratio to make the realized
            # loss-token fraction match the config.
            effective_weight = float(spec.weight)
            loss_ratio = None
            target_tokens = getattr(dataset, "num_target_tokens", None)
            input_tokens = getattr(dataset, "num_tokens", None)
            if mode == "sft" and target_tokens is not None and input_tokens:
                target_tokens = int(target_tokens)
                input_tokens = int(input_tokens)
                if target_tokens <= 0:
                    effective_weight = 0.0
                else:
                    loss_ratio = target_tokens / input_tokens
                    effective_weight = float(spec.weight) / max(loss_ratio, 1e-9)
            sources.append(
                BatchSource(
                    name=name,
                    factory=_batch_factory(
                        config,
                        mode,
                        dataset,
                        shard_rank=shard_rank,
                        shard_world_size=shard_world_size,
                        persistent_warmup=persistent_warmup,
                    ),
                    weight=effective_weight,
                )
            )
            meta_sources.append(
                {
                    "path": spec.path,
                    "weight": spec.weight,
                    "effective_weight": effective_weight,
                    "loss_token_ratio": loss_ratio,
                    "domain": spec.domain,
                    "tokens": getattr(dataset, "num_tokens", None),
                    "target_tokens": target_tokens,
                }
            )
        return (
            MixedDataset(meta={"sources": meta_sources}, datasets=list(child_datasets)),
            _mixer_factory(
                sources,
                total_batches=total_batches,
                seed=config.data.seed,
                shuffle=config.data.shuffle,
                batches_per_step=max(int(config.train.grad_accum), 1),
                lockstep=bool(config.data.lockstep),
            ),
        )
    if not path:
        raise SystemExit(
            "data.tokenized_path or data.sources is required; "
            "run `python -m minideepseek.data.prepare` first"
        )
    dataset = _load_single_dataset(mode, path, full_packs=train_full_packs)
    factory = _batch_factory(
        config,
        mode,
        dataset,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        persistent_warmup=persistent_warmup,
    )
    return dataset, factory



def _validate_dataset(dataset, tokenizer, config: TrainConfig, *, label: str) -> None:
    """Fail early on tokenizer/sequence-length drift between data and model."""
    for index, child in enumerate(getattr(dataset, "datasets", ())):
        _validate_dataset(child, tokenizer, config, label=f"{label}[{index}]")
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
    if isinstance(dataset, SFTDataset):
        if meta.get("format") != "minideepseek-sft-v1":
            raise ValueError(
                f"{label} SFT dataset has format {meta.get('format')!r}, expected "
                "'minideepseek-sft-v1'; rebuild it with the current "
                "minideepseek.data.sft builder"
            )
        tokens = int(dataset.num_tokens)
        targets = int(dataset.num_target_tokens)
        if not (0 < targets < tokens):
            raise ValueError(
                f"{label} SFT dataset has {targets} target tokens out of {tokens}; "
                "response-only masks must leave both input and target tokens"
            )
        offsets = np.asarray(dataset.offsets)
        if offsets.ndim != 1 or offsets.size < 2:
            raise ValueError(f"{label} SFT dataset has invalid offsets")
        lengths = np.diff(offsets)
        if lengths.size and int(lengths.min()) <= 0:
            raise ValueError(f"{label} SFT dataset contains an empty example")
        mask = np.asarray(dataset.target_mask)
        per_example = np.add.reduceat(mask.astype(np.int64), offsets[:-1])
        if int(per_example.min()) <= 0:
            raise ValueError(
                f"{label} SFT dataset has an example with zero target tokens; "
                "a response-only mask is broken"
            )


def _interleave_sources(
    sources: dict[str, Callable[[], Iterator[PackedBatch]]],
) -> Callable[[], Iterator[PackedBatch]]:
    """Round-robin the per-domain validation sets into one stream.

    ``Trainer.evaluate`` stops after ``train.val_steps`` batches, so a plain
    concatenation would only ever look at the first domain.  Each call rebuilds
    the iterators, which keeps every evaluation on the same deterministic
    prefix (the val factories use ``persistent_warmup=False``).
    """
    order = list(sources.values())

    def factory() -> Iterator[PackedBatch]:
        iterators = [iter(source()) for source in order]
        live = list(range(len(iterators)))
        while live:
            for index in list(live):
                try:
                    yield next(iterators[index])
                except StopIteration:
                    live.remove(index)

    return factory


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
        # A fresh, base-budget schedule keeps the validation batches identical
        # across evaluations (the training warmup must not leak into val).
        val_dataset, val_batches = load_dataset(
            config,
            mode,
            path=config.train.val_path,
            shard_rank=0,
            shard_world_size=1,
            persistent_warmup=False,
        )
        _validate_dataset(val_dataset, tokenizer, config, label="val")
    val_sources = {}
    for spec in config.train.val_sources:
        val_source, val_factory = load_dataset(
            config,
            mode,
            path=spec.path,
            shard_rank=0,
            shard_world_size=1,
            persistent_warmup=False,
        )
        _validate_dataset(val_source, tokenizer, config, label=f"val[{spec.domain}]")
        val_sources[spec.domain] = val_factory
    if val_batches is None and val_sources:
        # ``train.val_path`` is the single-set form; a curriculum only declares
        # ``train.val_sources``.  Without a fallback the trainer builds no
        # ``_val_factory`` and ``train.val_every`` silently does nothing, so the
        # whole run gets no validation signal.
        val_batches = _interleave_sources(val_sources)
    trainer, model = build_trainer(
        config,
        batches,
        val_batches=val_batches,
        val_sources=val_sources or None,
        tokenizer_hash=tokenizer.sha256,
        distributed=context,
    )
    summary = trainer.train()
    if summary.get("interrupted"):
        # The trainer already wrote step_<N>; leave ``final`` to a completed run.
        checkpoint = Path(config.train.out_dir) / f"step_{trainer.step:06d}"
    else:
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
    # The process group outlives this stage: a curriculum calls run() once per
    # stage in the same process, and re-initialising between stages fails
    # (2026-11-11).  ``init_distributed`` hands back the live context and the
    # process owns its lifetime; call ``context.shutdown()`` yourself if you
    # really need to tear it down.
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    parser.add_argument(
        "--model-config",
        default=None,
        help="model YAML (e.g. configs/model/mini_145m_a60m.yaml); "
        "overrides the config's model_file",
    )
    args = parser.parse_args(argv)
    config = load_config(TrainConfig, args.config, model_file=args.model_config)
    report = run(config, args.mode)
    # Only rank 0 prints: with N ranks the report would appear N times.
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
