"""Build a self-contained HTML training report from ``metrics.jsonl``.

The report is deliberately static: every chart is inline SVG and every asset
is embedded, so it can be committed as a run artifact or opened from a remote
machine without a plotting dependency.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .svg import PALETTE, Annotation, Chart

#: Keys shown in the summary cards, in display order.
_CARD_KEYS = (
    ("loss", "loss", "latest"),
    ("val_loss", "val loss", "latest"),
    ("val_ppl", "val ppl", "latest"),
    ("lr", "learning rate", "latest"),
    ("grad_norm", "grad norm", "latest"),
    ("tokens_per_sec", "throughput", "latest"),
    ("mem_peak_mb", "peak memory", "latest"),
    ("experts_used_fraction", "experts used", "latest"),
    ("routing_entropy_normalized", "router entropy", "latest"),
    ("swiglu_gate_clamp_rate", "SwiGLU clamp", "latest"),
)

_TABLE_KEYS = (
    "loss",
    "val_loss",
    "lr",
    "grad_norm",
    "tokens_per_sec",
    "mem_peak_mb",
    "experts_used_fraction",
)


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL metrics file, skipping blank/corrupt trailing lines."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid metrics JSON at {path}:{line_number}: {error}") from error
            if isinstance(record, dict):
                records.append(record)
    return records


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def latest(records: Sequence[dict[str, Any]], key: str) -> float | None:
    """Most recent finite value of ``key``."""
    for record in reversed(records):
        value = _number(record.get(key))
        if value is not None:
            return value
    return None


def _add_series(
    chart: Chart,
    records: Sequence[dict[str, Any]],
    key: str,
    label: str,
    *,
    color: str | None = None,
    kind: str = "line",
    x_key: str = "step",
    dash: str | None = None,
) -> bool:
    xs: list[float] = []
    ys: list[float] = []
    for record in records:
        x = _number(record.get(x_key))
        y = _number(record.get(key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    if not ys:
        return False
    chart.add_series(label, xs, ys, color=color, kind=kind, dash=dash)
    return True


def _has_key(records: Iterable[dict[str, Any]], key: str) -> bool:
    return any(_number(record.get(key)) is not None for record in records)


def _panel(title: str, ylabel: str, *, logy: bool = False, x_key: str = "step") -> Chart:
    return Chart(
        title=title,
        xlabel="step" if x_key == "step" else "tokens seen",
        ylabel=ylabel,
        logy=logy,
        width=640,
        height=280,
    )


def _loss_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("Loss", "cross-entropy")
    added = _add_series(chart, records, "loss", "total", color=PALETTE[0])
    added |= _add_series(chart, records, "loss_ema", "EMA", color="#60a5fa", dash="4 3")
    added |= _add_series(chart, records, "main_loss", "main head", color=PALETTE[2])
    if _has_key(records, "mtp_loss"):
        added |= _add_series(chart, records, "mtp_loss", "MTP", color=PALETTE[3])
    depths = sorted(
        int(key.rsplit("_d", 1)[1])
        for key in {key for record in records for key in record}
        if key.startswith("mtp_loss_d") and key.rsplit("_d", 1)[1].isdigit()
    )
    for depth in depths:
        added |= _add_series(
            chart, records, f"mtp_loss_d{depth}", f"MTP d{depth}", color=PALETTE[4]
        )
    if _has_key(records, "val_loss"):
        _add_series(chart, records, "val_loss", "validation", color="#111827", kind="points")
        added = True
    return chart if added else None


def _validation_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("Validation perplexity", "perplexity", logy=True)
    if not _add_series(chart, records, "val_ppl", "val ppl", color=PALETTE[1], kind="points"):
        return None
    return chart


def _lr_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("Learning rate", "lr", logy=True)
    return chart if _add_series(chart, records, "lr", "lr", color=PALETTE[0]) else None


def _grad_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("Gradient norm", "||g||", logy=True)
    return chart if _add_series(chart, records, "grad_norm", "grad norm", color=PALETTE[1]) else None


def _throughput_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("Throughput", "tokens / s")
    return (
        chart
        if _add_series(chart, records, "tokens_per_sec", "tokens/s", color=PALETTE[2])
        else None
    )


def _memory_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("Accelerator memory", "MiB")
    added = _add_series(chart, records, "mem_allocated_mb", "allocated", color=PALETTE[0])
    _add_series(chart, records, "mem_reserved_mb", "reserved", color=PALETTE[3])
    _add_series(chart, records, "mem_peak_mb", "peak", color=PALETTE[1], dash="4 3")
    return chart if added else None


def _router_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("MoE expert load", "max load / mean")
    added = _add_series(
        chart, records, "expert_load_max_over_mean", "mean over layers", color=PALETTE[0]
    )
    _add_series(
        chart, records, "expert_load_max_over_mean_worst", "worst layer", color=PALETTE[1]
    )
    _add_series(chart, records, "expert_usage_mean_std", "usage std", color=PALETTE[3])
    return chart if added else None


def _routing_health_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("Routing health", "fraction / normalized entropy")
    added = _add_series(
        chart, records, "routing_entropy_normalized", "entropy (1 = uniform)", color=PALETTE[2]
    )
    _add_series(chart, records, "experts_used_fraction", "experts used", color=PALETTE[4])
    return chart if added else None


def _swiglu_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("SwiGLU clamp rate", "clamped fraction")
    added = _add_series(
        chart, records, "swiglu_gate_clamp_rate", "gate", color=PALETTE[3]
    )
    _add_series(chart, records, "swiglu_up_clamp_rate", "up", color=PALETTE[5])
    return chart if added else None


def _aux_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("Auxiliary losses", "loss", logy=True)
    added = _add_series(chart, records, "indexer_loss", "indexer KL", color=PALETTE[4])
    _add_series(chart, records, "balance_loss", "balance", color=PALETTE[5])
    return chart if added else None


def _learning_curve_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    if not _has_key(records, "tokens_seen"):
        return None
    chart = _panel("Learning curve", "cross-entropy", x_key="tokens_seen")
    added = _add_series(chart, records, "loss", "train loss", color=PALETTE[0], x_key="tokens_seen")
    _add_series(
        chart,
        records,
        "val_loss",
        "val loss",
        color="#111827",
        kind="points",
        x_key="tokens_seen",
    )
    return chart if added else None


def _format_value(key: str, value: float | None) -> str:
    if value is None:
        return "—"
    if key in {"tokens_per_sec"}:
        if value >= 1e9:
            return f"{value / 1e9:.2f}G"
        if value >= 1e6:
            return f"{value / 1e6:.2f}M"
        if value >= 1e3:
            return f"{value / 1e3:.1f}k"
        return f"{value:.0f}"
    if key in {"mem_peak_mb", "mem_allocated_mb", "mem_reserved_mb"}:
        return f"{value / 1024:.2f} GiB"
    if key in {"experts_used_fraction", "routing_entropy_normalized", "swiglu_gate_clamp_rate"}:
        return f"{value * 100:.1f}%"
    if key == "lr":
        return f"{value:.2e}"
    if key == "grad_norm":
        return f"{value:.3f}"
    if key == "val_ppl":
        return f"{value:.1f}"
    return f"{value:.4f}"


def _cards(records: Sequence[dict[str, Any]]) -> str:
    cards = []
    for key, label, _ in _CARD_KEYS:
        value = latest(records, key)
        if value is None:
            continue
        cards.append(
            f'<div class="card"><div class="card-label">{html.escape(label)}</div>'
            f'<div class="card-value">{_format_value(key, value)}</div>'
            f'<div class="card-key">{html.escape(key)}</div></div>'
        )
    return "".join(cards)


def _summary_line(records: Sequence[dict[str, Any]]) -> str:
    steps = [record.get("step") for record in records if _number(record.get("step")) is not None]
    tokens = latest(records, "tokens_seen")
    parts = [f"{len(records)} log records"]
    if steps:
        parts.append(f"steps {int(min(steps))}–{int(max(steps))}")
    if tokens is not None:
        if tokens >= 1e9:
            parts.append(f"{tokens / 1e9:.2f}B tokens seen")
        elif tokens >= 1e6:
            parts.append(f"{tokens / 1e6:.1f}M tokens seen")
        else:
            parts.append(f"{tokens:,.0f} tokens seen")
    return " · ".join(parts)


def _table(records: Sequence[dict[str, Any]]) -> str:
    # Validation records share a step with the training record that produced
    # them; fold them into a ``val_loss`` column instead of printing half-empty
    # rows.
    val_by_step = {
        int(record["step"]): record["val_loss"]
        for record in records
        if _number(record.get("step")) is not None
        and _number(record.get("val_loss")) is not None
    }
    train_records = [record for record in records if _number(record.get("loss")) is not None]
    rows = []
    for record in train_records[-12:]:
        step = int(record.get("step", 0))
        cells = [f"<td>{step}</td>", f"<td>{_format_value('loss', _number(record.get('loss')))}</td>"]
        cells.append(f"<td>{_format_value('val_loss', _number(val_by_step.get(step)))}</td>")
        for key in _TABLE_KEYS[2:]:
            cells.append(f"<td>{_format_value(key, _number(record.get(key)))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{html.escape(key)}</th>" for key in ("step", *_TABLE_KEYS))
    return (
        "<table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 32px; background: #f8fafc; color: #0f172a;
       font-family: ui-sans-serif, -apple-system, "Segoe UI", sans-serif; }
header { max-width: 1400px; margin: 0 auto 20px; }
h1 { margin: 0 0 6px; font-size: 24px; letter-spacing: -0.02em; }
.sub { color: #64748b; font-size: 13px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.meta span { background: #eef2ff; color: #3730a3; border-radius: 999px;
             padding: 3px 10px; font-size: 12px; }
.cards { max-width: 1400px; margin: 0 auto 20px; display: grid;
         grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; }
.card-label { color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
.card-value { font-size: 22px; font-weight: 650; margin: 4px 0 2px; font-variant-numeric: tabular-nums; }
.card-key { color: #cbd5e1; font-size: 10px; font-family: ui-monospace, monospace; }
.grid { max-width: 1400px; margin: 0 auto; display: grid;
        grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px; }
.panel { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px; overflow: hidden; }
.notes { max-width: 1400px; margin: 20px auto 0; color: #64748b; font-size: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid #eef2f7; }
th:first-child, td:first-child { text-align: left; }
th { color: #64748b; font-weight: 600; background: #f8fafc; }
.table-wrap { max-width: 1400px; margin: 20px auto 0; background: #fff;
              border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; overflow-x: auto; }
@media (max-width: 560px) { body { padding: 16px; } .grid { grid-template-columns: 1fr; } }
"""


def build_report(
    records: Sequence[dict[str, Any]],
    out_path: str | Path,
    *,
    title: str = "miniDeepSeek training report",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a static HTML report and return its summary numbers."""
    if not records:
        raise ValueError("no metrics records to plot")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        _loss_panel(records),
        _learning_curve_panel(records),
        _validation_panel(records),
        _lr_panel(records),
        _grad_panel(records),
        _throughput_panel(records),
        _memory_panel(records),
        _router_panel(records),
        _routing_health_panel(records),
        _swiglu_panel(records),
        _aux_panel(records),
    ]
    rendered = [panel.render() for panel in panels if panel is not None]

    meta = dict(meta or {})
    meta_html = "".join(
        f"<span>{html.escape(str(key))}: {html.escape(str(value))}</span>"
        for key, value in meta.items()
        if value is not None
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="sub">{html.escape(_summary_line(records))}</div>
  <div class="meta">{meta_html}</div>
</header>
<section class="cards">{_cards(records)}</section>
<section class="grid">
{''.join(f'<div class="panel">{svg}</div>' for svg in rendered)}
</section>
<div class="table-wrap">{_table(records)}</div>
<div class="notes">Generated from metrics.jsonl · {len(rendered)} panels ·
latest step {int(latest(records, "step") or 0)}</div>
</body>
</html>
"""
    out_path.write_text(document, encoding="utf-8")
    return {
        "path": str(out_path),
        "records": len(records),
        "panels": len(rendered),
        "last_step": int(latest(records, "step") or 0),
        "last_loss": latest(records, "loss"),
        "last_val_loss": latest(records, "val_loss"),
        "last_val_ppl": latest(records, "val_ppl"),
        "tokens_seen": latest(records, "tokens_seen"),
    }


def add_chinchilla_marker(chart: Chart, tokens: float) -> Chart:
    """Add a vertical reference line at the Chinchilla token budget."""
    chart.annotations.append(Annotation(x=tokens, label="Chinchilla 20×"))
    return chart
