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

from .svg import PALETTE, Annotation, Chart, Heatmap

#: Summary cards, in display order.  Each entry lists fallback keys so a report
#: built from an older ``metrics.jsonl`` (flat, hash-mixed router keys) still
#: renders alongside one built after the hash/learned split.
_CARD_KEYS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("loss",), "loss"),
    (("val_loss",), "val loss"),
    (("val_ppl",), "val ppl"),
    (("lr",), "learning rate"),
    (("grad_norm",), "grad norm"),
    (("tokens_per_sec",), "throughput"),
    (("mem_peak_mb",), "peak memory"),
    (("eta_seconds",), "ETA"),
    (("index_topk",), "CSA top-k"),
    (("router_learned_min_relative_load",), "min load share"),
    (
        ("router_learned_routing_entropy_normalized", "routing_entropy_normalized"),
        "router entropy",
    ),
    (("swiglu_gate_clamp_rate",), "SwiGLU clamp"),
)

#: Rare 0/1 stability flags.  Plotted only when they fire, so a clean run shows
#: no panel at all instead of a flat line of zeros.
_STABILITY_KEYS = (
    ("loss_spike", "loss spike"),
    ("ar_active", "anticipatory AR"),
    ("rollback", "rollback"),
)

_TABLE_KEYS = (
    "loss",
    "val_loss",
    "lr",
    "grad_norm",
    "tokens_per_sec",
    "mem_peak_mb",
    "router_learned_min_relative_load",
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


def _first_present(records: Sequence[dict[str, Any]], keys: Sequence[str]) -> str | None:
    """First key in ``keys`` that carries a number somewhere in ``records``."""
    for key in keys:
        if _has_key(records, key):
            return key
    return None


def _add_series_any(
    chart: Chart,
    records: Sequence[dict[str, Any]],
    keys: Sequence[str],
    label: str,
    *,
    color: str | None = None,
    kind: str = "line",
    dash: str | None = None,
) -> bool:
    """Plot the first of ``keys`` that has data (new name, then legacy name)."""
    key = _first_present(records, keys)
    if key is None:
        return False
    return _add_series(chart, records, key, label, color=color, kind=kind, dash=dash)


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


def _val_domains(records: Sequence[dict[str, Any]]) -> list[str]:
    """Domain suffixes of the per-domain validation keys (``val_loss_<name>``)."""
    prefix = "val_loss_"
    names = {
        key[len(prefix) :]
        for record in records
        for key in record
        if key.startswith(prefix) and key[len(prefix) :]
    }
    return sorted(names)


def _domain_val_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    """Per-domain validation loss.

    The aggregate ``val_loss`` hides a regression that only hits one domain, so
    every ``val_loss_<domain>`` recorded by the trainer gets its own series here.
    """
    chart = _panel("Validation loss by domain", "cross-entropy")
    added = _add_series(chart, records, "val_loss", "all domains", color="#111827", dash="5 3")
    for index, name in enumerate(_val_domains(records)):
        added |= _add_series(
            chart, records, f"val_loss_{name}", name, color=PALETTE[index % len(PALETTE)]
        )
    return chart if added else None


def _lr_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("Learning rate", "lr", logy=True)
    return chart if _add_series(chart, records, "lr", "lr", color=PALETTE[0]) else None


def _schedule_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    """Schedules logged as fractions, so they share one 0–1 axis."""
    chart = _panel("Schedules", "fraction / weight")
    added = _add_series(chart, records, "mtp_weight", "MTP weight", color=PALETTE[3])
    added |= _add_series(chart, records, "lr_factor", "lr factor", color=PALETTE[0], dash="4 3")
    return chart if added else None


def _index_topk_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    """CSA sparse-attention budget: dense-over-compressed shrinking to top-k."""
    chart = _panel("CSA index top-k", "tokens attended")
    return (
        chart
        if _add_series(chart, records, "index_topk", "index top-k", color=PALETTE[4])
        else None
    )


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


def _timing_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    """Step wall time and remaining ETA.

    Log scale because the two live ~4 decades apart (seconds per step vs days
    remaining); on a linear axis the ETA would flatten the step time to nothing.
    """
    chart = _panel("Timing", "seconds", logy=True)
    added = _add_series(chart, records, "step_time_s", "step time", color=PALETTE[2])
    added |= _add_series(chart, records, "eta_seconds", "ETA", color=PALETTE[8], dash="4 3")
    return chart if added else None


def _memory_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("Accelerator memory", "MiB")
    added = _add_series(chart, records, "mem_allocated_mb", "allocated", color=PALETTE[0])
    _add_series(chart, records, "mem_reserved_mb", "reserved", color=PALETTE[3])
    _add_series(chart, records, "mem_peak_mb", "peak", color=PALETTE[1], dash="4 3")
    return chart if added else None


def _router_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    """Load balance for learned routers, with the hash layers kept separate.

    Hash layers route from a fixed ``tid2eid`` table, so their load can never be
    balanced and must not be averaged into the learned numbers.
    """
    chart = _panel("MoE expert load", "load / mean load")
    added = _add_series_any(
        chart,
        records,
        ("router_learned_expert_load_max_over_mean", "expert_load_max_over_mean"),
        "learned: max/mean",
        color=PALETTE[0],
    )
    added |= _add_series_any(
        chart,
        records,
        (
            "router_learned_expert_load_max_over_mean_worst",
            "expert_load_max_over_mean_worst",
        ),
        "learned: worst layer",
        color=PALETTE[1],
    )
    added |= _add_series_any(
        chart,
        records,
        ("router_learned_min_relative_load",),
        "learned: min share",
        color=PALETTE[2],
    )
    _add_series_any(
        chart,
        records,
        ("router_hash_expert_load_max_over_mean",),
        "hash (fixed table)",
        color=PALETTE[7],
        dash="4 3",
    )
    return chart if added else None


def _routing_health_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("Routing entropy", "normalized (1 = uniform)")
    added = _add_series_any(
        chart,
        records,
        ("router_learned_routing_entropy_normalized", "routing_entropy_normalized"),
        "learned (1 = uniform)",
        color=PALETTE[2],
    )
    _add_series_any(
        chart,
        records,
        ("router_hash_routing_entropy_normalized",),
        "hash (fixed table)",
        color=PALETTE[7],
        dash="4 3",
    )
    return chart if added else None


def _router_snapshots(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records that carry the raw per-router expert counts."""
    return [
        record
        for record in records
        if isinstance(record.get("router_layers"), dict) and record["router_layers"]
    ]


def _short_router_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) > 1 and parts[0] == "layers":
        return f"L{parts[1]}"
    if len(parts) > 1 and parts[0] == "mtp_layers":
        return f"MTP{parts[1]}"
    return name


def _router_layer_heatmap(records: Sequence[dict[str, Any]]) -> Heatmap | None:
    """Layer x Expert relative load, averaged over every snapshot."""
    snapshots = _router_snapshots(records)
    if not snapshots:
        return None
    order: list[str] = []
    totals: dict[str, list[float]] = {}
    hash_flags: dict[str, bool] = {}
    for record in snapshots:
        for name, layer in record["router_layers"].items():
            counts = layer.get("counts") or []
            if not counts:
                continue
            if name not in totals:
                order.append(name)
                totals[name] = [0.0] * len(counts)
                hash_flags[name] = bool(layer.get("hash", False))
            for index, value in enumerate(counts):
                totals[name][index] += float(value)
    names = [name for name in order if sum(totals[name]) > 0]
    if not names:
        return None
    values = []
    for name in names:
        counts = totals[name]
        total = sum(counts)
        mean = total / len(counts)
        values.append([count / mean for count in counts])
    return Heatmap(
        title=f"Expert load, Layer × Expert (mean of {len(snapshots)} snapshots)",
        row_labels=[
            f"{_short_router_name(name)} (hash)" if hash_flags.get(name) else _short_router_name(name)
            for name in names
        ],
        col_labels=[f"E{index}" for index in range(len(values[0]))],
        values=values,
        xlabel="expert · relative load (1.00 = balanced, colour saturates at ±0.5)",
        ylabel="router",
        center=1.0,
        min_span=0.5,
    )


def _domain_expert_heatmap(records: Sequence[dict[str, Any]]) -> Heatmap | None:
    """Domain x Expert enrichment over the learned routers.

    ``P(e|d) / P(e)``: 1.0 means the domain shows no preference for that expert,
    2.0 means it picks it twice as often as the global average.  Only learned
    routers take part -- hash routing is a function of the token id, so it would
    encode token statistics rather than a routing decision.
    """
    snapshots = _router_snapshots(records)
    if not snapshots:
        return None
    order: list[str] = []
    per_domain: dict[str, list[float]] = {}
    for record in snapshots:
        label = record.get("domain_name")
        if not isinstance(label, str) or not label:
            domain_id = _number(record.get("domain_id"))
            if domain_id is None or domain_id < 0:
                continue
            label = f"domain {int(domain_id)}"
        learned = [
            layer
            for layer in record["router_layers"].values()
            if not layer.get("hash") and (layer.get("counts") or [])
        ]
        if not learned:
            continue
        width = len(learned[0]["counts"])
        if label not in per_domain:
            order.append(label)
            per_domain[label] = [0.0] * width
        for layer in learned:
            for index, value in enumerate(layer["counts"]):
                per_domain[label][index] += float(value)
    if len(order) < 2:
        # A single domain has nothing to compare against; skip the panel.
        return None
    width = len(per_domain[order[0]])
    global_counts = [sum(per_domain[label][index] for label in order) for index in range(width)]
    global_total = sum(global_counts) or 1.0
    base = [count / global_total for count in global_counts]
    values = []
    for label in order:
        counts = per_domain[label]
        total = sum(counts) or 1.0
        values.append(
            [(counts[index] / total) / base[index] if base[index] else 0.0 for index in range(width)]
        )
    return Heatmap(
        title="Domain × Expert enrichment (learned routers)",
        row_labels=order,
        col_labels=[f"E{index}" for index in range(width)],
        values=values,
        xlabel="expert · P(e|d) / P(e)  (1.00 = no preference, colour saturates at ±0.5)",
        ylabel="domain",
        center=1.0,
        min_span=0.5,
        fmt="{:.2f}",
    )


def _router_bias_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    """How hard the auxiliary-loss-free balancer is pushing each expert.

    The bias itself is never plotted per expert (too many series); its spread and
    magnitude over time answer "is the balancer moving at all, and is it still
    able to move" -- a bias that saturates stops correcting.
    """
    snapshots = _router_snapshots(records)
    xs: list[float] = []
    spread: list[float] = []
    magnitude: list[float] = []
    for record in snapshots:
        biases = [
            layer["bias"]
            for layer in record["router_layers"].values()
            if isinstance(layer.get("bias"), list) and layer["bias"]
        ]
        step = _number(record.get("step"))
        if not biases or step is None:
            continue
        xs.append(step)
        spread.append(max(max(bias) - min(bias) for bias in biases))
        magnitude.append(max(max(abs(value) for value in bias) for bias in biases))
    if not xs:
        return None
    chart = _panel("Router balance bias", "bias units")
    chart.title = f"Router balance bias ({len(xs)} snapshots)"
    chart.add_series("max spread (max-min)", xs, spread, color=PALETTE[3])
    chart.add_series("max |bias|", xs, magnitude, color=PALETTE[4], dash="4 3")
    return chart


def _swiglu_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    chart = _panel("SwiGLU clamp rate", "clamped fraction")
    added = _add_series(
        chart, records, "swiglu_gate_clamp_rate", "gate", color=PALETTE[3]
    )
    _add_series(chart, records, "swiglu_up_clamp_rate", "up", color=PALETTE[5])
    return chart if added else None


def _swiglu_magnitude_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    """Largest raw SwiGLU activation against the configured ``swiglu_limit``.

    The clamp *rate* only says how often the limit binds; this shows how close
    activations run to it, which is what predicts a clamp-rate explosion.
    """
    chart = _panel("SwiGLU activation magnitude", "|activation| max")
    added = _add_series(chart, records, "swiglu_gate_abs_max", "gate", color=PALETTE[3])
    added |= _add_series(chart, records, "swiglu_up_abs_max", "up", color=PALETTE[5])
    return chart if added else None


def _stability_panel(records: Sequence[dict[str, Any]]) -> Chart | None:
    """Markers for the steps where a loss spike / AR / rollback actually fired."""
    chart = _panel("Stability events", "flag")
    fired_steps: set[float] = set()
    for index, (key, label) in enumerate(_STABILITY_KEYS):
        xs: list[float] = []
        ys: list[float] = []
        for record in records:
            x = _number(record.get("step"))
            value = _number(record.get(key))
            if x is None or value is None or value < 0.5:
                continue
            xs.append(x)
            ys.append(value)
        if not xs:
            continue
        fired_steps.update(xs)
        chart.add_series(label, xs, ys, color=PALETTE[1 + index], kind="points")
    if not fired_steps:
        # Nothing fired: a flat line at zero would only imply "we log this".
        return None
    # Count steps, not markers: one spike usually fires both spike and AR.
    count = len(fired_steps)
    chart.title = f"Stability events ({count} step{'s' if count > 1 else ''})"
    return chart


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


#: Keys rendered as a 0-1 fraction; matched by suffix so both the flat legacy
#: names and the ``router_learned_*`` / ``router_hash_*`` split names work.
_PERCENT_SUFFIXES = (
    "experts_used_fraction",
    "routing_entropy_normalized",
    "min_relative_load",
    "clamp_rate",
)


def _format_duration(seconds: float) -> str:
    """Compact duration, e.g. ``45s`` / ``12m`` / ``3.4h`` / ``2.1d``.

    Kept local instead of importing ``train.metrics``: that module pulls in
    torch, and ``viz`` is meant to render metrics without a plotting or GPU
    dependency.
    """
    if not math.isfinite(seconds) or seconds < 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _format_value(key: str, value: float | None) -> str:
    if value is None:
        return "—"
    if key == "eta_seconds":
        return _format_duration(value)
    if key == "index_topk":
        return f"{value:.0f}"
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
    if key.endswith(_PERCENT_SUFFIXES):
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
    for candidates, label in _CARD_KEYS:
        key = _first_present(records, candidates)
        if key is None:
            continue
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
        _domain_val_panel(records),
        _validation_panel(records),
        _lr_panel(records),
        _schedule_panel(records),
        _grad_panel(records),
        _throughput_panel(records),
        _timing_panel(records),
        _memory_panel(records),
        _router_panel(records),
        _routing_health_panel(records),
        _router_layer_heatmap(records),
        _domain_expert_heatmap(records),
        _router_bias_panel(records),
        _index_topk_panel(records),
        _swiglu_panel(records),
        _swiglu_magnitude_panel(records),
        _aux_panel(records),
        _stability_panel(records),
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
        "val_domains": _val_domains(records),
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
