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


def test_format_metrics_leads_with_the_main_head_loss():
    """``loss`` folds in MTP; the console line must lead with the LM loss.

    ``loss = main + 0.3 * mtp``, so a step whose MTP head is doing badly shows a
    headline ~0.85 nats above the language-modelling number and moving for
    reasons that have nothing to do with the LM.  Watching it made a healthy
    run look flat.
    """
    line = format_metrics(
        12,
        {
            "loss": 3.4219,
            "loss_ema": 3.5,
            "main_loss": 2.5567,
            "main_loss_ema": 2.61,
            "mtp_loss": 2.8840,
            "mtp_weight": 0.3,
            "lr": 2.4e-4,
            "lr_factor": 0.5,
            "grad_norm": 0.83,
            "tokens_per_sec": 12_345,
            "mem_peak_mb": 2048,
            "eta_seconds": 7200,
        },
    )
    assert "step" in line
    assert line.startswith("step     12 | lm 2.5567 (ema 2.6100)")
    assert "total 3.4219 (mtp 2.8840 x0.3)" in line
    assert "lr 2.40e-04 x0.500" in line
    assert "12.35k tok/s" in line
    assert "2.0 GiB peak" in line
    assert "ETA 2.0h" in line


def test_format_metrics_shows_per_domain_training_losses_in_mix_order():
    line = format_metrics(
        7,
        {
            "main_loss": 2.4,
            "train_main_loss_general": 2.5512,
            "train_main_loss_code": 1.4803,
            "train_main_loss_tool": 1.0211,
        },
    )
    # general is 82% of the mix and must not sort behind code.
    assert "general 2.5512 code 1.4803 tool 1.0211" in line


def test_format_metrics_shows_per_domain_validation():
    line = format_metrics(
        2400,
        {
            "val_loss": 3.46,
            "val_main_loss": 1.8402,
            "val_ppl": 6.30,
            "val_main_loss_general": 2.5901,
            "val_main_loss_code": 1.4402,
            "val_main_loss_tool": 1.1801,
        },
    )
    assert "VAL lm 1.8402 ppl 6.30" in line
    assert "general 2.5901 code 1.4402 tool 1.1801" in line


def test_metrics_logger_writes_jsonl_and_echo(tmp_path, capsys):
    logger = MetricsLogger(tmp_path)
    logger.log(3, {"loss": 1.5, "tokens_per_sec": 1000})
    payload = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert payload["step"] == 3
    assert payload["loss"] == 1.5
    output = capsys.readouterr().out
    assert "step      3" in output
    assert "total 1.5000" in output
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


def test_dashboard_slims_per_domain_training_losses_and_keeps_snapshots_out_of_rows():
    """The live dashboard must forward the per-domain curves it draws.

    ``router_domain_layers`` is ~1 KB per snapshot step and the page only ever
    renders the newest one, so it may not ride along inside ``rows``.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "live_dashboard", root / "scripts" / "live_dashboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    record = {
        "step": 5250,
        "main_loss": 2.04,
        "train_main_loss_general": 2.24,
        "train_main_loss_code": 1.13,
        "train_main_loss_tool": 0.846,
        "lr_factor": 0.79,
        "router_domain_general_js_vs_pooled": 0.01,
        "router_layers": {"layers.0.feed_forward.router": {"counts": [1, 2]}},
        "router_domain_layers": {"overall": {"layers.0.feed_forward.router": {"counts": [1, 2]}}},
    }
    slim = module._slim(record)
    assert slim["train_main_loss_general"] == 2.24
    assert slim["train_main_loss_tool"] == 0.846
    assert slim["router_domain_general_js_vs_pooled"] == 0.01
    assert "router_domain_layers" not in slim
    assert "router_layers" not in slim

    best = module._best_val(
        [
            {"step": 100, "val_main_loss": 1.9},
            {"step": 200, "val_main_loss": 1.7},
            {"step": 300, "val_main_loss": 1.8},
            {"step": 400},
        ]
    )
    assert best == {"value": 1.7, "step": 200}
    assert module._best_val([{"step": 1}]) is None


def test_router_series_is_per_mille_and_padded_to_the_step_axis():
    """The trend panel indexes every domain/layer by snapshot position."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "live_dashboard_series", root / "scripts" / "live_dashboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def snapshot(step, counts_by_domain):
        return {
            "step": step,
            "domains": {
                name: {"L2": {"counts": counts}}
                for name, counts in counts_by_domain.items()
            },
        }

    series = module._router_series(
        [
            snapshot(100, {"general": [50, 50], "overall": [50, 50]}),
            # ``tool`` is missing here on purpose: 3% of the mix means most
            # single windows have none.
            snapshot(150, {"general": [80, 20], "overall": [80, 20]}),
        ]
    )
    assert series["steps"] == [100, 150]
    assert series["domains"] == ["overall", "general"]
    assert series["layers"] == ["L2"]
    assert series["shares"]["general"]["L2"] == [[500, 500], [800, 200]]
    for domain in series["shares"].values():
        for rows in domain.values():
            assert len(rows) == len(series["steps"])
