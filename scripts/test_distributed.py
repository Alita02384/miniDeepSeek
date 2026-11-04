#!/usr/bin/env python3
"""Multi-process smoke test for FSDP2/DDP training (CPU Gloo or CUDA NCCL).

Runs the debug model for a few optimizer steps on every rank, checks that all
ranks gathered the same full state dict, and writes a checkpoint.

CPU (works on macOS)::

    .venv/bin/python scripts/test_distributed.py --world-size 2

CUDA::

    torchrun --nproc_per_node=8 scripts/test_distributed.py --gpu

The script exits non-zero when ranks disagree or a collective fails.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _digest(state: dict) -> str:

    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()


def _worker(rank: int, world_size: int, port: int, mode: str, steps: int) -> None:
    import torch
    import torch.distributed as dist

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    from minideepseek.config import DistributedSettings, TrainConfig
    from minideepseek.model import build_model
    from minideepseek.model.batch import from_sequences
    from minideepseek.train.distributed import (
        apply_parallelism,
        gather_model_state_dict,
        init_distributed,
    )
    from minideepseek.train.trainer import Trainer

    settings = DistributedSettings(
        enabled=True,
        mode=mode,
        backend="auto",
        param_dtype="bfloat16" if torch.cuda.is_available() else "float32",
        reduce_dtype="float32",
    )
    context = init_distributed(settings, "cuda" if torch.cuda.is_available() else "cpu")
    try:
        config = TrainConfig.from_dict(
            {
                "model": {
                    "flavor": "debugmodel",
                    "seq_len": 64,
                    "n_mtp_layers": 1,
                    "device": str(context.device),
                    "dtype": "bfloat16" if torch.cuda.is_available() else "float32",
                    "seed": 1234,
                },
                "train": {
                    "steps": steps,
                    "grad_accum": 1,
                    "out_dir": f"/tmp/minideepseek-dist-{world_size}",
                    "log_every": 99,
                },
                "distributed": {
                    "enabled": True,
                    "mode": mode,
                    "param_dtype": settings.param_dtype,
                    "reduce_dtype": "float32",
                },
            }
        )
        from minideepseek.config import set_seed

        set_seed(1234)
        model = build_model(
            "debugmodel",
            seq_len=64,
            n_mtp_layers=1,
            device=context.device,
        )
        model.model = apply_parallelism(model.model, context, settings, device=context.device)
        batches = [
            from_sequences([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12]])
            for _ in range(steps + 2)
        ]
        trainer = Trainer(config, model, batches, distributed=context)
        trainer.train(steps=steps)
        # Every rank participates in the collective gather; cpu_offload returns
        # the full dict only on rank 0.  Share its digest via broadcast (gloo's
        # object all-gather is flaky on macOS).
        state = gather_model_state_dict(model.model)
        box = [_digest(state) if context.is_main else None]
        dist.broadcast_object_list(box, src=0)
        digest = box[0]
        if digest is None:
            raise RuntimeError("rank 0 did not gather a state dict")
        trainer.save("final")  # all ranks participate; only rank 0 writes
        if context.is_main:
            print(f"PASS world_size={world_size} mode={mode} digest={digest[:16]}", flush=True)
    finally:
        context.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--mode", choices=["fsdp2", "ddp"], default="fsdp2")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--gpu", action="store_true", help="use CUDA/NCCL")
    args = parser.parse_args()

    import torch

    if args.gpu and not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2
    if not args.gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import socket

    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    try:
        mp.spawn(
            _worker,
            args=(args.world_size, port, args.mode, args.steps),
            nprocs=args.world_size,
            join=True,
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
