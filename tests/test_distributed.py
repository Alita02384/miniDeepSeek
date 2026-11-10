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
    load_optimizer_state_dict,
)
from minideepseek.train.optimizer import build_optimizer
from minideepseek.train.spike import restore, snapshot
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
    # At least one sub-optimizer must have state after resume.
    assert any(optimizer.state for optimizer in other.optimizer.optimizers)
    # The restored optimizer state must be on the same layout as the FSDP2
    # parameters; otherwise the next step mixes Tensor and DTensor state.
    other.train_step()


def test_distributed_snapshot_restore(distributed_context):
    context, settings = distributed_context
    model = build_model("debugmodel", seq_len=64, device="cpu")
    model.model = apply_parallelism(model.model, context, settings, device=context.device)
    optimizer = build_optimizer(model, TrainConfig().optim)
    optimizer.zero_grad()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = snapshot(7, model.model, optimizer, distributed=context)
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    restore(state, model.model, optimizer, distributed=context)
    after = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_load_optimizer_state_from_single_process_checkpoint():
    """A run without a distributed context saves ``optimizer.state_dict()``,
    which keys state by integer position instead of parameter name.  Loading
    it under ``distributed`` (a 1-GPU run continued on 2 GPUs) must still find
    every state tensor."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(8, 4)).sum().backward()
    optimizer.step()
    plain = optimizer.state_dict()
    assert plain["state"] and all(isinstance(key, int) for key in plain["state"])

    restored = torch.optim.AdamW(model.parameters(), lr=1e-3)
    load_optimizer_state_dict(model, restored, plain)
    assert len(restored.state) == len(optimizer.state)
    for position, values in optimizer.state.items():
        assert torch.equal(restored.state[position]["exp_avg"], values["exp_avg"])


def test_init_distributed_reuses_a_live_process_group(monkeypatch):
    """A curriculum runs stage after stage in one process; re-initialising the
    group between stages fails, so the live context is handed back instead."""
    from minideepseek.train import distributed as D

    cached = D.DistributedContext(enabled=True, rank=0, world_size=2, local_rank=0)
    monkeypatch.setattr(D, "_CACHED", cached)
    monkeypatch.setattr(D.dist, "is_available", lambda: True)
    monkeypatch.setattr(D.dist, "is_initialized", lambda: True)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    def explode(*args, **kwargs):
        raise AssertionError("init_process_group must not run twice")

    monkeypatch.setattr(D.dist, "init_process_group", explode)
    context = D.init_distributed(DistributedSettings(mode="ddp"), "cpu")
    assert context is cached
