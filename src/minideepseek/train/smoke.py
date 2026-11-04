"""Stage-1 smoke test: forward, backward, optimizer, checkpoint round-trip.

Run with the V4 debug model (the project's CI model)::

    python -m minideepseek.train.smoke --flavor debugmodel --steps 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..config import (
    ModelSettings,
    OptimizerSettings,
    ScheduleSettings,
    TrainConfig,
    TrainSettings,
    resolve_device,
)
from ..data.pretrain import PackedPretrainDataset
from ..data.sft import SFTDataset
from ..model.batch import PackedBatch, from_sequences
from ..model.checkpoint import load_checkpoint
from ..model.wrapper import build_model
from ..tokenizer import load_tokenizer
from ..train.trainer import build_trainer


def synthetic_batches(
    *, vocab_size: int, num_batches: int, seq_len: int, seed: int = 0
) -> list[PackedBatch]:
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        length = max(8, seq_len // 4)
        docs = []
        for _ in range(2):
            doc_len = max(4, length // 2)
            docs.append(torch.randint(3, vocab_size, (doc_len,), generator=generator).tolist())
        batches.append(from_sequences(docs))
    return batches


def build_batches(args, vocab_size: int):
    if args.data:
        path = Path(args.data)
        if args.data_kind == "sft":
            dataset = SFTDataset(path)
            return [dataset[i] for i in range(min(len(dataset), args.steps * 2))], dataset.meta
        dataset = PackedPretrainDataset(path)
        return [dataset[i] for i in range(min(len(dataset), args.steps * 2))], dataset.meta
    return synthetic_batches(
        vocab_size=vocab_size,
        num_batches=max(args.steps * 2, 2),
        seq_len=args.seq_len,
        seed=args.seed,
    ), {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", default="debugmodel")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-mtp-layers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--out-dir", default="checkpoints/smoke")
    parser.add_argument("--data")
    parser.add_argument("--data-kind", choices=["pretrain", "sft"], default="pretrain")
    parser.add_argument("--index-topk", type=int, default=None, help="CSA top-k override (debug default: none)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    tokenizer = load_tokenizer()
    batches, data_meta = build_batches(args, 2048 if args.flavor == "debugmodel" else 6400)
    if len(batches) < args.steps * args.grad_accum:
        raise SystemExit("not enough batches for the requested steps")

    config = TrainConfig(
        model=ModelSettings(
            flavor=args.flavor,
            seq_len=args.seq_len,
            n_mtp_layers=args.n_mtp_layers,
            swiglu_limit=10.0,
            index_topk=args.index_topk,
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
        ),
        optim=OptimizerSettings(lr=args.lr),
        schedule=ScheduleSettings(warmup_steps=1, decay="cosine", min_lr_ratio=0.1),
        train=TrainSettings(
            steps=args.steps,
            grad_accum=args.grad_accum,
            log_every=1,
            out_dir=args.out_dir,
            mtp_weight=0.3,
            balance_loss_coeff=1e-4,
        ),
    )
    trainer, model = build_trainer(
        config, batches, tokenizer_hash=tokenizer.sha256
    )
    summary = trainer.train(args.steps)
    checkpoint = trainer.save("final")
    report = {
        "flavor": args.flavor,
        "params": model.parameter_report(),
        "summary": summary,
        "checkpoint": str(checkpoint),
        "data": data_meta,
    }
    if not args.no_verify:
        reloaded = build_model(
            args.flavor,
            seq_len=args.seq_len,
            n_mtp_layers=args.n_mtp_layers,
            device=device,
            swiglu_limit=10.0,
            index_topk=args.index_topk,
        )
        load_checkpoint(checkpoint, reloaded)
        batch = batches[0].to(device)
        with torch.no_grad():
            original = model.forward(batch, use_mtp=False)[0]
            restored = reloaded.forward(batch, use_mtp=False)[0]
        max_diff = float((original - restored).abs().max())
        report["checkpoint_max_abs_diff"] = max_diff
        report["checkpoint_verified"] = max_diff == 0.0
        if max_diff != 0.0:
            raise SystemExit(f"checkpoint round-trip mismatch: {max_diff}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
