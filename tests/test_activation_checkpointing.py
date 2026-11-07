"""Activation checkpointing must be numerically transparent.

The 8K curriculum stage only fits on 24 GB cards with per-block recompute, so the
switch has to be exact: same loss, same gradients, just more compute.
"""

from __future__ import annotations

import torch

from minideepseek.config import ModelSettings, TrainConfig, TrainSettings
from minideepseek.model import build_model, from_sequences
from minideepseek.model import model as model_module
from minideepseek.train.trainer import build_trainer


def _build(*, checkpointing: bool, n_mtp_layers: int = 1, seed: int = 0):
    torch.manual_seed(seed)
    return build_model(
        "debugmodel",
        seq_len=64,
        n_mtp_layers=n_mtp_layers,
        device="cpu",
        dtype="float32",
        swiglu_limit=10.0,
        activation_checkpointing=checkpointing,
    )


def _grads(model):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.model.named_parameters()
        if parameter.grad is not None
    }


def _one_step(model, sequences):
    model.train()
    output = model.compute_loss(from_sequences(sequences), mtp_weight=0.3)
    output.total.backward()
    return output.total.detach(), _grads(model)


def test_checkpointing_matches_plain_gradients():
    sequences = [[5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15]]
    plain = _build(checkpointing=False)
    recomputed = _build(checkpointing=True)
    for (name_a, param_a), (name_b, param_b) in zip(
        plain.model.named_parameters(), recomputed.model.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)

    loss_plain, grads_plain = _one_step(plain, sequences)
    loss_recomputed, grads_recomputed = _one_step(recomputed, sequences)

    # CPU reductions are thread-order dependent, so compare numerically rather
    # than bitwise; fp32 gradients must still agree to rounding error.
    torch.testing.assert_close(loss_recomputed, loss_plain, rtol=1e-6, atol=1e-6)
    assert grads_plain.keys() == grads_recomputed.keys()
    for name, grad in grads_plain.items():
        torch.testing.assert_close(
            grads_recomputed[name], grad, rtol=1e-5, atol=1e-6, msg=name
        )


def test_checkpointing_only_active_in_training(monkeypatch):
    calls = {"count": 0}
    real = model_module._checkpoint

    def spy(*args, **kwargs):
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(model_module, "_checkpoint", spy)
    model = _build(checkpointing=True)

    model.eval()
    with torch.no_grad():
        model.compute_loss(from_sequences([[5, 6, 7, 8, 9, 10]]), mtp_weight=0.3)
    assert calls["count"] == 0

    model.train()
    output = model.compute_loss(from_sequences([[5, 6, 7, 8, 9, 10]]), mtp_weight=0.3)
    output.total.backward()
    # 4 transformer blocks + 1 MTP block, once per forward.
    assert calls["count"] == 5


def test_trainer_plumbs_the_config_flag(tmp_path):
    config = TrainConfig(
        model=ModelSettings(
            flavor="debugmodel",
            seq_len=64,
            n_mtp_layers=1,
            device="cpu",
            dtype="float32",
            seed=0,
        ),
        train=TrainSettings(
            steps=1,
            out_dir=str(tmp_path / "run"),
            activation_checkpointing=True,
        ),
    )
    trainer, model = build_trainer(config, [from_sequences([[5, 6, 7, 8, 9]])])
    assert model.model.grad_checkpointing is True
    summary = trainer.train(1)
    assert summary["steps_run"] == 1
    assert summary["last_loss"] is not None
