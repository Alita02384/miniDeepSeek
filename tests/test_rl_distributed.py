"""CPU FSDP2 coverage for GRPO/OPD (world_size=1 Gloo)."""

from __future__ import annotations

import socket

import pytest
import torch

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


def _config(tmp_path, distributed: DistributedSettings, **rl):
    return RLConfig(
        model=ModelSettings(
            flavor="debugmodel",
            seq_len=128,
            n_mtp_layers=0,
            device="cpu",
            dtype="float32",
            seed=0,
        ),
        optim=OptimizerSettings(lr=1e-4),
        schedule=ScheduleSettings(warmup_steps=0, decay="constant", min_lr_ratio=1.0),
        train=TrainSettings(steps=1, log_every=1, out_dir=str(tmp_path / "rl")),
        rl=RLSettings(group_size=2, prompts_per_step=2, max_new_tokens=3, **rl),
        reward=RewardSettings(kind="math"),
        distributed=distributed,
    )


def test_fsdp2_grpo_one_step(tmp_path, distributed_context, stub_tokenizer):
    context, settings = distributed_context
    records = [
        GRPORecord(prompt=[{"role": "user", "content": "q"}], domain="math"),
    ]
    config = _config(tmp_path, settings)
    trainer, model = build_grpo_trainer(
        config,
        records,
        lambda text, _record: RewardResult(total=0.0),
        tokenizer=stub_tokenizer,
        distributed=context,
    )
    summary = trainer.train(1)
    assert summary["steps_run"] == 1
    path = trainer.save("final")
    assert (path / "model.pt").exists()
    assert torch.isfinite(torch.tensor(summary.get("loss_last", 0.0)))


def test_fsdp2_opd_one_step(tmp_path, distributed_context, stub_tokenizer):
    context, settings = distributed_context
    prompts = [OPDPrompt(prompt=[{"role": "user", "content": "q"}], teacher_key="math")]
    teacher = load_teacher(
        TeacherSettings(name="math", flavor="debugmodel", seq_len=128, dtype="float32"),
        tokenizer=stub_tokenizer,
        device="cpu",
    )
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.add_(0.01)
    config = _config(tmp_path, settings)
    config.rl.teacher_weights = {"math": 1.0}
    trainer, student = build_opd_trainer(
        config,
        prompts,
        tokenizer=stub_tokenizer,
        teacher_models={"math": teacher},
        distributed=context,
    )
    summary = trainer.train(1)
    assert summary["steps_run"] == 1
    path = trainer.save("final")
    assert (path / "model.pt").exists()
    assert torch.isfinite(torch.tensor(summary.get("loss_last", 0.0)))
