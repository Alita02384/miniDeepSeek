"""CPU (Gloo) coverage for the distributed path.

Real multi-GPU runs use NCCL; these tests exercise the same FSDP2 wrapping,
optimizer, and checkpoint-collective code on a single CPU process so the code
path cannot rot.  A 2-process smoke test lives in ``scripts/test_distributed.py``
because pytest-forked subprocesses are awkward on macOS.
"""

from __future__ import annotations

import socket

import pytest
import torch

from minideepseek.config import DistributedSettings, TrainConfig
from minideepseek.model import build_model
from minideepseek.model.batch import from_sequences
from minideepseek.train.distributed import (
    apply_parallelism,
    gather_model_state_dict,
    init_distributed,
)
from minideepseek.train.optimizer import build_optimizer
from minideepseek.train.trainer import Trainer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture()
def distributed_context(monkeypatch):
    port = _free_port()
    for name, value in {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
    }.items():
        monkeypatch.setenv(name, value)
    settings = DistributedSettings(
        enabled=True,
        mode="fsdp2",
        backend="gloo",
        param_dtype="float32",
        reduce_dtype="float32",
    )
    context = init_distributed(settings, "cpu")
    try:
        yield context, settings
    finally:
        context.shutdown()


def test_fsdp2_forward_backward_and_gather(distributed_context):
    context, settings = distributed_context
    model = build_model("debugmodel", seq_len=64, device="cpu")
    model.model = apply_parallelism(model.model, context, settings, device=context.device)
    batch = from_sequences([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12]])
    loss = model.compute_loss(batch)
    loss.total.backward()
    optimizer = build_optimizer(model, TrainConfig().optim)
    optimizer.step()
    state = gather_model_state_dict(model.model)
    assert state["tok_embeddings.weight"].shape == (2048, 256)
    assert "layers.0.attention.wq_a.weight" in state
    assert torch.isfinite(loss.total)


def test_fsdp2_trainer_save_and_resume(tmp_path, distributed_context):
    context, settings = distributed_context
    batches = [
        from_sequences([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12]])
        for _ in range(8)
    ]
    config = TrainConfig.from_dict(
        {
            "model": {
                "flavor": "debugmodel",
                "seq_len": 64,
                "n_mtp_layers": 1,
                "device": "cpu",
                "dtype": "float32",
                "seed": 1,
            },
            "train": {
                "steps": 1,
                "grad_accum": 2,
                "out_dir": str(tmp_path / "run"),
                "log_every": 1,
            },
            "distributed": {
                "enabled": True,
                "mode": "fsdp2",
                "backend": "gloo",
                "param_dtype": "float32",
                "reduce_dtype": "float32",
            },
        }
    )
    model = build_model(
        "debugmodel", seq_len=64, n_mtp_layers=1, device="cpu"
    )
    model.model = apply_parallelism(model.model, context, settings, device=context.device)
    trainer = Trainer(config, model, batches, distributed=context)
    trainer.train(steps=1)
    checkpoint = trainer.save("final")
    assert (checkpoint / "model.pt").exists()
    assert (checkpoint / "meta.json").exists()

    resumed = build_model("debugmodel", seq_len=64, n_mtp_layers=1, device="cpu")
    resumed.model = apply_parallelism(
        resumed.model, context, settings, device=context.device
    )
    other = Trainer(config, resumed, batches, distributed=context)
    meta = other.resume(checkpoint)
    assert meta["step"] == 1
