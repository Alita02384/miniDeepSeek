#!/usr/bin/env python3
"""Benchmark training throughput and peak memory for one model configuration.

The measured loop is the real training path (forward + MTP/indexer losses,
backward, grad clip, optimizer step), so ``tokens/s`` is directly comparable
with the trainer.  ``tokens/s`` counts main-head prediction targets.

Examples::

    # quick single-device sanity check
    python scripts/benchmark.py --model-config configs/model/mini_220m_a80m.yaml \
        --seq-len 512 --batch-size 1 --steps 5 --warmup 1 --device cuda

    # multi-GPU FSDP2 (one process per GPU)
    torchrun --nproc_per_node=8 scripts/benchmark.py --config configs/pretrain/full_6b.yaml \
        --distributed fsdp2 --batch-size 1 --steps 10

    # machine-readable output
    python scripts/benchmark.py --flavor mini_145m_a60m --json-out bench.json \
        --markdown-out doc/benchmark_results.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minideepseek.config import (  # noqa: E402
    DistributedSettings,
    ModelSettings,
    TrainConfig,
    load_config,
    resolve_device,
)
from minideepseek.model import build_model, from_sequences  # noqa: E402
from minideepseek.train.benchmark import BenchmarkResult, run_benchmark  # noqa: E402
from minideepseek.train.distributed import (  # noqa: E402
    apply_parallelism,
    init_distributed,
)
from minideepseek.train.optimizer import build_optimizer  # noqa: E402

_MARKDOWN_HEADER = (
    "| model | params | active | batch | seq | step ms | tok/s | peak mem |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _load_train_config(args: argparse.Namespace) -> TrainConfig:
    if args.config:
        return load_config(TrainConfig, args.config, model_file=args.model_config)
    if args.model_config:
        return TrainConfig(model=load_config(ModelSettings, args.model_config))
    flavor = args.flavor or "mini_220m_a80m"
    path = ROOT / "configs" / "model" / f"{flavor}.yaml"
    if not path.exists():
        raise SystemExit(f"unknown flavor {flavor!r}; expected {path}")
    return TrainConfig(model=load_config(ModelSettings, path))


def _batch_factory(settings: ModelSettings, batch_size: int, seq_len: int, device: torch.device):
    generator = torch.Generator().manual_seed(0)
    vocab = max(int(settings.vocab_size), 8)

    def factory():
        sequences = [
            torch.randint(3, vocab, (seq_len,), generator=generator).tolist()
            for _ in range(batch_size)
        ]
        return from_sequences(sequences, device=device)

    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="training YAML (model + optimizer)")
    parser.add_argument("--model-config", default=None, help="model YAML")
    parser.add_argument("--flavor", default=None, help="bundled model flavor")
    parser.add_argument("--label", default=None, help="row label in the report")
    parser.add_argument("--batch-size", type=int, default=1, help="documents per step")
    parser.add_argument("--seq-len", type=int, default=None, help="tokens per document")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None, help="float32 | bfloat16")
    parser.add_argument("--distributed", choices=["none", "fsdp2", "ddp"], default="none")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None, help="append a markdown row to this file")
    parser.add_argument("--mtp-weight", type=float, default=0.3)
    args = parser.parse_args(argv)

    config = _load_train_config(args)
    if args.seq_len is not None:
        config.model.seq_len = args.seq_len
        config.model.max_context = max(config.model.max_context, args.seq_len)
    if args.dtype is not None:
        config.model.dtype = args.dtype
    if args.device != "auto":
        config.model.device = args.device

    context = None
    settings: DistributedSettings | None = None
    if args.distributed != "none":
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise SystemExit(
                "--distributed requires torchrun (RANK/WORLD_SIZE are not set)"
            )
        settings = DistributedSettings(
            enabled=True,
            mode=args.distributed,
            backend="auto",
            param_dtype=(
                "bfloat16"
                if torch.cuda.is_available() and config.model.dtype == "bfloat16"
                else "float32"
            ),
            reduce_dtype="float32",
        )
        context = init_distributed(settings, config.model.device)
        device = context.device
    else:
        device = resolve_device(config.model.device)

    dtype = config.model.dtype if device.type == "cuda" else "float32"
    model = build_model(config.model, device=device, dtype=dtype)
    if context is not None and settings is not None:
        model.model = apply_parallelism(
            model.model, context, settings, device=device
        )
    optimizer = build_optimizer(model, config.optim)
    seq_len = int(config.model.seq_len)
    label = args.label or Path(
        args.config or args.model_config or f"{args.flavor or 'mini_220m_a80m'}.yaml"
    ).stem
    result: BenchmarkResult = run_benchmark(
        model,
        optimizer,
        _batch_factory(config.model, args.batch_size, seq_len, device),
        steps=args.steps,
        warmup=args.warmup,
        device=device,
        grad_clip=config.optim.grad_clip,
        mtp_weight=args.mtp_weight,
        label=label,
    )
    world_size = context.world_size if context is not None else 1
    result.extra = {
        "dtype": dtype,
        "distributed": args.distributed,
        "world_size": world_size,
        "seq_len": seq_len,
        "per_rank_tokens_per_sec": result.tokens_per_sec,
        "per_rank_peak_mem_mb": result.peak_mem_mb,
    }
    # Every data-parallel rank consumes its own batch, so the run-level
    # throughput is the per-rank number times the world size.
    result.tokens_per_sec *= world_size
    result.tokens_per_step *= world_size

    if context is None or context.is_main:
        payload = result.as_dict()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(result.markdown_row())
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        if args.markdown_out:
            path = Path(args.markdown_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            prefix = "" if path.exists() else _MARKDOWN_HEADER
            with path.open("a", encoding="utf-8") as handle:
                handle.write(prefix + result.markdown_row() + "\n")

    if context is not None:
        context.barrier()
        context.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
