"""Validation plumbing: a curriculum only declares ``train.val_sources``.

Before this, ``Trainer._val_factory`` was built from ``val_batches`` alone, and
``run()`` only filled ``val_batches`` from ``train.val_path``.  A curriculum
(every ``configs/pretrain/generated/curriculum_*.yaml``) therefore ran with
``_val_factory is None`` and ``train.val_every`` never fired -- 11k steps with
no validation signal at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from minideepseek.config import (
    ModelSettings,
    OptimizerSettings,
    ScheduleSettings,
    TrainConfig,
    TrainSettings,
)
from minideepseek.model import from_sequences
from minideepseek.train.run import _interleave_sources
from minideepseek.train.trainer import build_trainer


def _config(tmp_path: Path, **train_overrides) -> TrainConfig:
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


def test_interleave_sources_round_robins_and_terminates():
    def source(tag: str, count: int):
        def factory():
            for index in range(count):
                yield f"{tag}{index}"

        return factory

    mixed = list(_interleave_sources({"a": source("a", 2), "b": source("b", 3)})())
    # a0, b0, a1, b1, then b2 (a is exhausted) -- never a plain concatenation.
    assert mixed == ["a0", "b0", "a1", "b1", "b2"]
    # Each call rebuilds the iterators, so evaluations stay reproducible.
    assert list(_interleave_sources({"a": source("a", 2), "b": source("b", 3)})()) == mixed


def test_trainer_logs_validation_from_val_sources_only(tmp_path):
    """The exact curriculum shape: val sources, no ``train.val_path``."""
    batch = from_sequences([[1, 2, 3, 4, 5, 6], [7, 8, 9]])
    config = _config(tmp_path, val_every=1, val_steps=2)
    val_factory = _interleave_sources(
        {
            "general": lambda: iter([batch, batch]),
            "code": lambda: iter([batch]),
        }
    )
    trainer, model = build_trainer(
        config,
        [batch, batch],
        val_batches=val_factory,
        val_sources=None,
        tokenizer_hash="stub",
    )
    assert trainer._val_factory is not None  # noqa: SLF001 - the regression itself
    trainer.train(2)
    metrics = sorted((tmp_path / "run").rglob("metrics.jsonl"))
    assert metrics, "no metrics.jsonl under the run directory"
    records = [
        json.loads(line)
        for line in metrics[-1].read_text().splitlines()
        if line.strip()
    ]
    val_records = [record for record in records if "val_loss" in record]
    assert len(val_records) == 2, records
    assert 0.0 < val_records[-1]["val_loss"] < 100.0
    assert val_records[-1]["val_tokens"] > 0
    assert model.training is True  # evaluate() must restore train mode
