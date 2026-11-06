from __future__ import annotations

import torch

from minideepseek.config import OptimizerSettings
from minideepseek.model import from_sequences
from minideepseek.train.benchmark import run_benchmark
from minideepseek.train.optimizer import build_optimizer


def test_run_benchmark_measures_target_tokens(make_debug_model):
    model = make_debug_model(n_mtp_layers=0, seq_len=8)
    optimizer = build_optimizer(model, OptimizerSettings(lr=1e-3))
    batch = from_sequences([[1, 2, 3, 4, 5, 6, 7, 8]])

    result = run_benchmark(
        model,
        optimizer,
        lambda: batch,
        steps=1,
        warmup=0,
        device=torch.device("cpu"),
        mtp_weight=0.0,
        label="debug",
    )

    assert result.label == "debug"
    assert result.steps == 1
    assert result.docs_per_batch == 1
    assert result.seq_len == 8
    # One document of eight tokens yields seven prediction targets.
    assert result.tokens_per_step == 7
    assert result.tokens_per_sec > 0
    assert result.step_time_ms > 0
    assert result.params_total > 0
    assert result.peak_mem_mb is None  # CPU has no accelerator counter
