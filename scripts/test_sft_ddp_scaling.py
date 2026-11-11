#!/usr/bin/env python3
"""2-process regression for the SFT token-scaling bug (audit section 3).

Two ranks use the *same* debug model and the same packed batch, but a different
number of response-only target tokens.  The step must produce bit-identical
weights on both ranks: the trainer has to divide the reduced gradient by the
*global* target count, not each rank's local count.

    .venv/bin/python scripts/test_sft_ddp_scaling.py --dist-mode ddp
    .venv/bin/python scripts/test_sft_ddp_scaling.py --dist-mode fsdp2
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TARGET_MASKS = {
    0: [False, True, True, True, True, False, False, False],
    1: [False, True, True, False, False, False, False, False],
}


def _worker(rank: int, world: int, port: int, dist_mode: str) -> None:
    import torch

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    from minideepseek.config import (
        DistributedSettings,
        ModelSettings,
        OptimizerSettings,
        ScheduleSettings,
        TrainConfig,
        TrainSettings,
    )
    from minideepseek.model.batch import from_sequences
    from minideepseek.train.distributed import (
        gather_model_state_dict,
        init_distributed,
    )
    from minideepseek.train.trainer import build_trainer

    settings = DistributedSettings(
        enabled=True,
        mode=dist_mode,
        backend="gloo",
        param_dtype="float32",
        reduce_dtype="float32",
    )
    context = init_distributed(settings, "cpu")
    try:
        config = TrainConfig(
            model=ModelSettings(
                flavor="debugmodel",
                seq_len=64,
                n_mtp_layers=0,
                swiglu_limit=10.0,
                device="cpu",
                dtype="float32",
                seed=0,
            ),
            optim=OptimizerSettings(
                kind="adamw", lr=1e-2, weight_decay=0.0, grad_clip=1e9
            ),
            schedule=ScheduleSettings(
                warmup_steps=0, decay="constant", min_lr_ratio=1.0
            ),
            train=TrainSettings(
                steps=1,
                grad_accum=1,
                log_every=1,
                out_dir=f"/tmp/minideepseek-sft-scale-{dist_mode}-rank{rank}",
                balance_loss_coeff=0.0,
                mtp_weight=0.0,
            ),
            distributed=settings,
        )
        batch = from_sequences(
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            target_masks=[TARGET_MASKS[rank]],
        )
        trainer, model = build_trainer(
            config,
            [batch],
            tokenizer_hash="stub",
            distributed=context,
        )
        local_targets = int(TARGET_MASKS[rank].count(True))
        if rank == 0:
            print(f"targets rank0={local_targets}", flush=True)
        else:
            print(f"targets rank1={local_targets}", flush=True)
        trainer.train_step()
        state = {
            name: tensor.detach().float().cpu()
            for name, tensor in gather_model_state_dict(
                model.model, all_ranks=True
            ).items()
        }
        gathered: list[dict | None] = [None for _ in range(world)]
        torch.distributed.all_gather_object(gathered, state)
        if rank == 0:
            reference = gathered[0]
            assert reference is not None
            max_diff = 0.0
            for other in gathered[1:]:
                assert other is not None
                assert set(other) == set(reference)
                for name, value in reference.items():
                    max_diff = max(
                        max_diff,
                        float((value - other[name]).abs().max().item()),
                    )
            print(f"max weight diff = {max_diff:.3e}", flush=True)
            if max_diff > 1e-7:
                raise AssertionError(
                    "ranks diverged after one step: the gradient was not scaled "
                    "by the global token count"
                )
            print(f"PASS sft-{dist_mode} world={world}", flush=True)
    finally:
        context.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--dist-mode", choices=["ddp", "fsdp2"], default="ddp")
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    import torch.multiprocessing as mp

    try:
        mp.spawn(
            _worker,
            args=(args.world_size, port, args.dist_mode),
            nprocs=args.world_size,
            join=True,
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
