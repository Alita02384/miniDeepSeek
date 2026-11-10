#!/usr/bin/env python3
"""2-process CPU smoke test for FSDP2 GRPO/OPD (Gloo).

    OMP_NUM_THREADS=1 .venv/bin/python scripts/test_rl_distributed.py --method grpo

On CUDA use the same script with ``--device cuda`` (NCCL) or launch it under
torchrun; it is a correctness smoke, not a throughput benchmark.
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


def _worker(
    rank: int,
    world: int,
    port: int,
    method: str,
    device: str,
    dist_mode: str,
    rollout_backend: str,
) -> None:
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
        RewardSettings,
        RLConfig,
        RLSettings,
        ScheduleSettings,
        TeacherSettings,
        TrainSettings,
    )
    from minideepseek.data.grpo import GRPORecord
    from minideepseek.data.opd import OPDPrompt
    from minideepseek.rl.grpo import build_grpo_trainer
    from minideepseek.rl.opd import build_opd_trainer, load_teacher
    from minideepseek.rl.reward import RewardResult
    from minideepseek.train.distributed import init_distributed

    class StubTokenizer:
        vocab_size = 2048
        sha256 = "stub-tokenizer"
        eos_token_id = 2
        bos_token_id = 1
        pad_token_id = 0
        stop_token_ids = [2, 0]

        def render_chat(self, messages, *, add_generation_prompt=False, tools=None, open_thinking=False):
            del add_generation_prompt, tools, open_thinking
            return " ".join(str(message.get("content", "")) for message in messages)

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            ids = [3 + (ord(char) % 2044) for char in str(text)][:64]
            return ids or [3]

        def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            return " ".join(str(int(value)) for value in ids)

    settings = DistributedSettings(
        enabled=True,
        mode=dist_mode,
        backend="gloo" if device == "cpu" else "nccl",
        param_dtype="float32" if device == "cpu" else "bfloat16",
        reduce_dtype="float32",
    )
    context = init_distributed(settings, device)
    tokenizer = StubTokenizer()
    try:
        config = RLConfig(
            model=ModelSettings(
                flavor="debugmodel",
                seq_len=128,
                n_mtp_layers=0,
                device=device,
                dtype="float32" if device == "cpu" else "bfloat16",
                seed=0,
            ),
            optim=OptimizerSettings(lr=1e-4),
            schedule=ScheduleSettings(warmup_steps=0, decay="constant", min_lr_ratio=1.0),
            train=TrainSettings(
                steps=1,
                log_every=99,
                out_dir=f"/tmp/minideepseek-rl-dist-{method}",
            ),
            rl=RLSettings(
                group_size=2,
                prompts_per_step=2,
                max_new_tokens=3,
                rollout_backend=rollout_backend,
            ),
            reward=RewardSettings(kind="math"),
            distributed=settings,
        )
        if method == "grpo":
            records = [
                GRPORecord(prompt=[{"role": "user", "content": str(index)}], domain="math")
                for index in range(world)
            ]
            trainer, _ = build_grpo_trainer(
                config,
                records,
                lambda text, _record: RewardResult(total=0.0),
                tokenizer=tokenizer,
                distributed=context,
            )
        else:
            prompts = [
                OPDPrompt(
                    prompt=[{"role": "user", "content": str(index)}], teacher_key="math"
                )
                for index in range(world)
            ]
            teacher = load_teacher(
                TeacherSettings(name="math", flavor="debugmodel", seq_len=128, dtype="float32"),
                tokenizer=tokenizer,
                device=device,
            )
            with torch.no_grad():
                for parameter in teacher.parameters():
                    parameter.add_(0.01)
            config.rl.teacher_weights = {"math": 1.0}
            trainer, _ = build_opd_trainer(
                config,
                prompts,
                tokenizer=tokenizer,
                teacher_models={"math": teacher},
                distributed=context,
            )
        summary = trainer.train(1)
        path = trainer.save("final")
        if rank == 0:
            print(
                f"PASS {method} world={world} step={summary['steps_run']} ckpt={path}",
                flush=True,
            )
    finally:
        context.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["grpo", "opd"], default="grpo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dist-mode", choices=["fsdp2", "ddp"], default="fsdp2")
    parser.add_argument("--rollout-backend", choices=["torch", "cached"], default="torch")
    args = parser.parse_args()
    if args.device != "cpu":
        import torch

        if not torch.cuda.is_available():
            print("CUDA not available", file=sys.stderr)
            return 2
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    import torch.multiprocessing as mp

    try:
        mp.spawn(
            _worker,
            args=(
                args.world_size,
                port,
                args.method,
                args.device,
                args.dist_mode,
                args.rollout_backend,
            ),
            nprocs=args.world_size,
            join=True,
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
