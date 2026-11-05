from __future__ import annotations

import json

import torch

from minideepseek.config import (
    AnticipatoryRoutingSettings,
    ModelSettings,
    OptimizerSettings,
    ScheduleSettings,
    TrainConfig,
    TrainSettings,
)
from minideepseek.data.sft import SFTDataset
from minideepseek.data.sft import iter_batches as sft_batches
from minideepseek.model import from_sequences
from minideepseek.train.losses import iter_moe_layers, update_expert_biases
from minideepseek.train.trainer import build_trainer


def _config(tmp_path, **train_overrides) -> TrainConfig:
    train = dict(
        steps=2,
        grad_accum=1,
        log_every=1,
        out_dir=str(tmp_path / "run"),
        mtp_weight=0.3,
        balance_loss_coeff=1e-4,
    )
    train.update(train_overrides)
    return TrainConfig(
        model=ModelSettings(
            flavor="debugmodel",
            seq_len=128,
            n_mtp_layers=1,
            swiglu_limit=10.0,
            device="cpu",
            dtype="float32",
            seed=0,
        ),
        optim=OptimizerSettings(lr=1e-3),
        schedule=ScheduleSettings(warmup_steps=1, decay="cosine", min_lr_ratio=0.1),
        train=TrainSettings(**train),
    )


def test_trainer_two_steps_and_checkpoint(tmp_path):
    batches = [from_sequences([[1, 2, 3, 4, 5, 6], [7, 8, 9]]), from_sequences([[10, 11, 12, 13, 14]])]
    config = _config(tmp_path)
    trainer, model = build_trainer(config, batches, tokenizer_hash="stub")
    summary = trainer.train(2)
    assert summary["steps_run"] == 2
    assert summary["last_loss"] is not None
    assert (tmp_path / "run" / "metrics.jsonl").exists()
    metrics = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(metrics) == 2
    assert metrics[-1]["tokens"] > 0
    assert metrics[-1]["swiglu_calls"] > 0
    checkpoint = trainer.save("final")
    assert (checkpoint / "meta.json").exists()
    # The checkpoint must reproduce logits exactly.
    from minideepseek.model import build_model
    from minideepseek.model.checkpoint import load_checkpoint

    reloaded = build_model("debugmodel", seq_len=128, n_mtp_layers=1, device="cpu", swiglu_limit=10.0)
    load_checkpoint(checkpoint, reloaded)
    batch = batches[0]
    with torch.no_grad():
        assert torch.equal(model.forward(batch, use_mtp=False)[0], reloaded.forward(batch, use_mtp=False)[0])


def test_gradient_accumulation(tmp_path):
    batch = from_sequences([[1, 2, 3, 4, 5, 6]])
    config = _config(tmp_path, steps=1, grad_accum=2)
    trainer, _model = build_trainer(config, [batch, batch], tokenizer_hash="stub")
    summary = trainer.train(1)
    assert summary["tokens"] == 2 * (int(batch.target_mask.sum()) + max(int(batch.target_mask.sum()) - 1, 0))


def test_sft_trainer(tmp_path):
    import numpy as np

    from minideepseek.data.io import save_flat_dataset

    dataset_path = tmp_path / "sft.npz"
    input_ids = np.asarray([1, 10, 11, 12, 2], dtype=np.int32)
    save_flat_dataset(
        dataset_path,
        input_ids=input_ids,
        positions=np.arange(len(input_ids), dtype=np.int32),
        target_mask=np.asarray([False, False, True, True, True], dtype=bool),
        offsets=np.asarray([0, len(input_ids)], dtype=np.int64),
        meta={"format": "test"},
    )
    dataset = SFTDataset(dataset_path)
    config = _config(tmp_path, steps=1)
    loader = sft_batches(dataset, batch_tokens=256, shuffle=False)
    trainer, _model = build_trainer(config, loader, tokenizer_hash="stub")
    summary = trainer.train(1)
    assert summary["steps_run"] == 1


def test_expert_bias_update(debug_model, small_batch):
    debug_model.train()
    with torch.no_grad():
        outputs = debug_model.forward(small_batch, use_mtp=False)
        assert outputs
    layers = list(iter_moe_layers(debug_model.model))
    learned = [(name, moe) for name, moe in layers if not moe.router.hash]
    hashed = [(name, moe) for name, moe in layers if moe.router.hash]
    assert learned and hashed
    biases_before = [moe.expert_bias_E.detach().clone() for _name, moe in learned]
    assert any(int(moe.router.tokens_per_expert_E.sum().item()) > 0 for _name, moe in hashed)
    updated = update_expert_biases(debug_model.model)
    assert updated == len(learned)
    for (_name, moe), before in zip(learned, biases_before):
        assert not torch.equal(moe.expert_bias_E, before)
    for _name, moe in layers:
        # Counters are per-step and must be cleared on every layer.
        assert torch.equal(
            moe.router.tokens_per_expert_E,
            torch.zeros_like(moe.router.tokens_per_expert_E),
        )


def test_hash_layers_have_no_expert_bias(debug_model):
    buffer_names = dict(debug_model.model.named_buffers())
    for name, moe in iter_moe_layers(debug_model.model):
        if moe.router.hash:
            assert moe.expert_bias_E is None
            assert f"{name}.expert_bias_E" not in buffer_names
        else:
            assert moe.expert_bias_E is not None
            assert f"{name}.expert_bias_E" in buffer_names


def test_anticipatory_routing_trainer(tmp_path):
    batch = from_sequences([[1, 2, 3, 4, 5], [6, 7, 8]])
    config = _config(tmp_path, steps=2)
    config.anticipatory = AnticipatoryRoutingSettings(enabled=True, delta=1)
    trainer, model = build_trainer(config, [batch, batch, batch], tokenizer_hash="stub")
    summary = trainer.train(2)
    assert summary["steps_run"] == 2
    assert trainer._ar_state is not None
    assert trainer._ar_state.stale_snapshot() is not None
