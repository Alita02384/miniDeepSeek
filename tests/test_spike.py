from __future__ import annotations

import pytest
import torch

from minideepseek.config import (
    ModelSettings,
    OptimizerSettings,
    ScheduleSettings,
    TrainConfig,
    TrainSettings,
)
from minideepseek.model import build_model, from_sequences
from minideepseek.train.scheduler import IndexTopKSchedule
from minideepseek.train.spike import SpikeDetector, restore, snapshot
from minideepseek.train.trainer import build_trainer


def test_spike_detector_flags_outlier():
    detector = SpikeDetector(window=20, sigma=3.0, min_history=5)
    for value in [1.0, 1.02, 0.99, 1.01, 1.0, 1.03, 0.98]:
        assert not detector.update(value)
    assert detector.update(10.0)
    assert detector.spikes == 1
    assert not detector.update(1.0)


def test_rollback_snapshot_restores_weights(debug_model):
    optimizer = torch.optim.AdamW(debug_model.parameters(), lr=1e-3)
    saved = snapshot(3, debug_model, optimizer)
    with torch.no_grad():
        next(debug_model.parameters()).add_(1.0)
    step = restore(saved, debug_model, optimizer)
    assert step == 3
    restored = build_model(
        "debugmodel", seq_len=128, n_mtp_layers=0, device="cpu", swiglu_limit=10.0
    )
    restored.load_state_dict(saved.model_state)
    for left, right in zip(debug_model.parameters(), restored.parameters()):
        assert torch.equal(left, right)


def test_index_topk_schedule():
    schedule = IndexTopKSchedule(start=1024, final=64, warmup_ratio=0.2)
    assert schedule.topk_at(0, 100) == 1024
    assert schedule.topk_at(79, 100) == 1024
    assert schedule.topk_at(100, 100) == 64
    middle = schedule.topk_at(90, 100)
    assert 64 < middle < 1024
    passive = IndexTopKSchedule(start=1024, final=64, warmup_ratio=0.0)
    assert passive.topk_at(99, 100) == 1024


def test_index_topk_schedule_visits_a_bounded_ladder():
    schedule = IndexTopKSchedule(start=2048, final=64, start_step=100, end_step=1100)
    ladder = {schedule.topk_at(step, 10_000) for step in range(0, 1201)}

    # Powers of two between final and start: 6 specializations, not ~1000.
    assert ladder == {64, 128, 256, 512, 1024, 2048}
    assert [schedule.topk_at(step, 10_000) for step in (0, 1100, 1200)] == [2048, 64, 64]


def test_index_topk_schedule_explicit_window():
    schedule = IndexTopKSchedule(start=2048, final=64, start_step=100, end_step=200)
    assert schedule.topk_at(0, 1000) == 2048
    assert schedule.topk_at(100, 1000) == 2048
    # Quantised to powers of two: every new top-k is a Triton specialization.
    assert schedule.topk_at(150, 1000) == 1024
    assert schedule.topk_at(200, 1000) == 64
    assert schedule.topk_at(999, 1000) == 64
    with pytest.raises(ValueError, match="greater than start_step"):
        IndexTopKSchedule(start=8, final=4, start_step=10, end_step=10)
    with pytest.raises(ValueError, match="set together"):
        IndexTopKSchedule(start=8, final=4, start_step=10)


def test_set_index_topk_on_debug_model(debug_model):
    original = debug_model.current_index_topk()
    assert original is not None
    debug_model.set_index_topk(3)
    assert debug_model.current_index_topk() == 3
    debug_model.set_index_topk(original)
    assert debug_model.current_index_topk() == original


def test_trainer_applies_index_schedule(tmp_path):
    batch = from_sequences([[1, 2, 3, 4, 5, 6]])
    config = TrainConfig(
        model=ModelSettings(
            flavor="debugmodel",
            seq_len=128,
            n_mtp_layers=0,
            swiglu_limit=10.0,
            device="cpu",
            dtype="float32",
            seed=0,
        ),
        optim=OptimizerSettings(lr=1e-3),
        schedule=ScheduleSettings(warmup_steps=0, decay="constant", min_lr_ratio=1.0),
        train=TrainSettings(
            steps=4,
            log_every=1,
            out_dir=str(tmp_path / "run"),
            index_topk_final=2,
            index_topk_warmup_ratio=0.5,
        ),
    )
    trainer, model = build_trainer(config, [batch, batch], tokenizer_hash="stub")
    assert model.current_index_topk() == 16
    metrics = trainer.train_step()
    assert metrics["index_topk"] == 16.0
    trainer.step = 1
    metrics = trainer.train_step()
    assert metrics["index_topk"] == 16.0
    trainer.step = 3
    metrics = trainer.train_step()
    assert metrics["index_topk"] < 16.0
    assert model.current_index_topk() < 16


def test_trainer_applies_explicit_index_window(tmp_path):
    batch = from_sequences([[1, 2, 3, 4, 5, 6]])
    config = TrainConfig(
        model=ModelSettings(
            flavor="debugmodel",
            seq_len=128,
            n_mtp_layers=0,
            swiglu_limit=10.0,
            device="cpu",
            dtype="float32",
            seed=0,
        ),
        optim=OptimizerSettings(lr=1e-3),
        schedule=ScheduleSettings(warmup_steps=0, decay="constant", min_lr_ratio=1.0),
        train=TrainSettings(
            steps=5,
            log_every=1,
            out_dir=str(tmp_path / "run"),
            index_topk_final=2,
            index_topk_start_step=1,
            index_topk_end_step=3,
        ),
    )
    trainer, model = build_trainer(config, [batch] * 5, tokenizer_hash="stub")
    assert model.current_index_topk() == 16
    metrics = trainer.train_step()
    assert metrics["index_topk"] == 16.0
    # Past the end of the explicit window the model stays at the sparse value.
    trainer.step = 4
    metrics = trainer.train_step()
    assert metrics["index_topk"] == 2.0


def test_spike_auto_enables_anticipatory_routing(tmp_path):
    batch = from_sequences([[1, 2, 3, 4, 5, 6]])
    config = TrainConfig(
        model=ModelSettings(
            flavor="debugmodel",
            seq_len=128,
            n_mtp_layers=0,
            swiglu_limit=10.0,
            device="cpu",
            dtype="float32",
            seed=0,
        ),
        optim=OptimizerSettings(lr=1e-3),
        schedule=ScheduleSettings(warmup_steps=0, decay="constant", min_lr_ratio=1.0),
        train=TrainSettings(
            steps=10,
            log_every=1,
            out_dir=str(tmp_path / "run"),
            loss_spike_window=10,
            loss_spike_min_history=3,
            loss_spike_auto_ar=True,
            loss_spike_ar_steps=5,
        ),
    )
    trainer, _model = build_trainer(config, [batch], tokenizer_hash="stub")
    trainer.step = 7
    for value in (1.0, 1.0, 1.0):
        trainer.spike_detector.update(value)
    record = trainer._handle_spike(100.0)
    assert record["loss_spike"] == 1.0
    assert trainer._ar_force_until == 12
    assert trainer._ar_enabled()
