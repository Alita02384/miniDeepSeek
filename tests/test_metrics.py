from __future__ import annotations

import json

from minideepseek.train.metrics import (
    MetricsLogger,
    format_count,
    format_duration,
    format_metrics,
    memory_stats,
)


def test_format_count_and_duration():
    assert format_count(999) == "999"
    assert format_count(1500) == "1.50k"
    assert format_count(2_500_000) == "2.50M"
    assert format_duration(45) == "45s"
    assert format_duration(120) == "2m"
    assert format_duration(7200) == "2.0h"


def test_format_metrics_is_human_readable():
    line = format_metrics(
        12,
        {
            "loss": 3.4219,
            "loss_ema": 3.5,
            "val_loss": 3.1,
            "val_ppl": 22.2,
            "lr": 2.4e-4,
            "grad_norm": 0.83,
            "tokens_per_sec": 12_345,
            "mem_peak_mb": 2048,
            "eta_seconds": 7200,
        },
    )
    assert "step" in line
    assert "loss 3.4219" in line
    assert "ema 3.5000" in line
    assert "val 3.1000" in line
    assert "ppl 22.2" in line
    assert "12.35k tok/s" in line
    assert "2.0 GiB peak" in line
    assert "ETA 2.0h" in line


def test_metrics_logger_writes_jsonl_and_echo(tmp_path, capsys):
    logger = MetricsLogger(tmp_path)
    logger.log(3, {"loss": 1.5, "tokens_per_sec": 1000})
    payload = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert payload["step"] == 3
    assert payload["loss"] == 1.5
    output = capsys.readouterr().out
    assert "step      3" in output
    assert "loss 1.5000" in output
    assert "1.00k tok/s" in output


def test_memory_stats_on_cpu_is_empty():
    assert memory_stats("cpu") == {}


def test_logger_summary_tolerates_sparse_records(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log(1, {"loss": 2.0}, echo=False)
    logger.log(1, {"val_loss": 1.0}, echo=False)
    summary = logger.summary()
    assert summary["loss_last"] == 2.0
    assert summary["val_loss_last"] == 1.0
