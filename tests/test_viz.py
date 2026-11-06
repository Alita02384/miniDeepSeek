from __future__ import annotations

import math

import pytest

from minideepseek.viz import Chart, build_report, nice_ticks
from minideepseek.viz.svg import Annotation


def _records(count: int = 20) -> list[dict]:
    records = []
    for step in range(count):
        records.append(
            {
                "step": step,
                "loss": 5.0 * math.exp(-step / 5) + 2.0,
                "loss_ema": 5.0 * math.exp(-step / 6) + 2.0,
                "lr": 1e-3 * (1 - step / count),
                "grad_norm": 1.0 + 0.1 * step,
                "tokens_per_sec": 1000 + 10 * step,
                "tokens_seen": (step + 1) * 1024,
                "expert_load_max_over_mean": 1.2,
                "routing_entropy_normalized": 0.98,
                "swiglu_gate_clamp_rate": 0.0,
            }
        )
        if step % 5 == 0:
            records.append(
                {
                    "step": step,
                    "val_loss": 2.5 - step * 0.01,
                    "val_ppl": math.exp(2.5 - step * 0.01),
                    "tokens_seen": (step + 1) * 1024,
                }
            )
    return records


def test_nice_ticks_cover_range():
    ticks = nice_ticks(0, 10)
    assert ticks[0] == 0
    assert ticks[-1] == 10
    assert len(ticks) >= 4


def test_chart_renders_polyline_and_ticks():
    chart = Chart(title="loss", ylabel="loss")
    chart.add_series("train", list(range(5)), [5, 4, 3, 2, 1])
    svg = chart.render()
    assert svg.startswith("<svg")
    assert "<polyline" in svg
    assert ">loss<" in svg
    assert svg.count("<text") >= 4


def test_chart_log_scale_skips_nonpositive_values():
    chart = Chart(title="lr", logy=True)
    chart.add_series("lr", [1, 2, 3], [0.0, 1e-4, 1e-3])
    svg = chart.render()
    assert "<polyline" in svg
    assert "1.0e-04" in svg or "1e-04" in svg


def test_chart_empty_data_is_rendered_as_placeholder():
    svg = Chart(title="empty").render()
    assert "no data" in svg


def test_chart_annotation_near_right_edge_uses_end_anchor():
    chart = Chart(title="scaling", logx=True, logy=True)
    chart.add_series("run", [1e6, 1e7, 1e8, 1e9], [5.0, 4.5, 4.1, 3.9])
    chart.annotations.append(Annotation(x=1e9, label="Chinchilla"))
    svg = chart.render()
    assert 'text-anchor="end"' in svg


def test_build_report_writes_self_contained_html(tmp_path):
    out = tmp_path / "report.html"
    summary = build_report(_records(), out, title="demo", meta={"model": "debug"})
    document = out.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert "<svg" in document
    assert "demo" in document
    assert "model: debug" in document
    assert summary["records"] == len(_records())
    assert summary["panels"] >= 5


def test_build_report_rejects_empty_records(tmp_path):
    with pytest.raises(ValueError, match="no metrics"):
        build_report([], tmp_path / "report.html")
