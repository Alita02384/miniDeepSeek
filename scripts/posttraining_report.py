#!/usr/bin/env python3
"""Merge post-training metrics and render a pretraining-style HTML report.

Each stage writes its own ``metrics.jsonl`` (SFT under ``runs/<run_id>/``,
GRPO/OPD at the stage root), so the run history is split across seven files
with different schemas.  This script:

* discovers ``sft_*``, ``grpo_*`` and ``opd_*`` phase directories;
* merges every record, keeping the original ``step`` and adding ``phase``,
  ``phase_kind``, ``phase_domain``, ``phase_step`` and ``global_step``;
* writes ``<root>/metrics_all.jsonl`` and ``<root>/summary.json``;
* writes a self-contained HTML report (inline SVG, no plotting dependency).

Usage::

    python scripts/posttraining_report.py checkpoints/posttrain_e2e_4k_145m
    python scripts/posttraining_report.py <root> --out reports/posttrain.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minideepseek.viz.svg import PALETTE, Chart  # noqa: E402

PHASE_RE = re.compile(r"^(sft|grpo|opd)_(.+)$")
KIND_ORDER = {"sft": 0, "grpo": 1, "opd": 2}
KIND_TITLE = {"sft": "SFT", "grpo": "GRPO", "opd": "OPD"}
DOMAIN_COLOR = {
    "general": PALETTE[0],
    "code": PALETTE[1],
    "tool": PALETTE[2],
    "mixed": PALETTE[4],
}
KIND_COLOR = {"sft": PALETTE[0], "grpo": PALETTE[3], "opd": PALETTE[4]}
DOMAIN_ORDER = {"general": 0, "code": 1, "tool": 2, "mixed": 3}

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 28px 34px 48px; font: 14px/1.5 -apple-system,
  BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  color: #1f2937; background: #f8fafc; }
header { margin-bottom: 22px; }
h1 { margin: 0 0 4px; font-size: 25px; letter-spacing: -0.02em; }
h2 { margin: 30px 0 10px; font-size: 18px; }
h3 { margin: 0 0 2px; font-size: 14px; font-weight: 600; }
.sub { color: #64748b; font-size: 13px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 10px;
  color: #475569; font-size: 12px; }
.meta span { background: #fff; border: 1px solid #e2e8f0; border-radius: 6px;
  padding: 3px 8px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
  gap: 10px; margin: 18px 0 8px; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 9px;
  padding: 12px 14px; }
.card .label { color: #64748b; font-size: 11px; text-transform: uppercase;
  letter-spacing: .04em; }
.card .value { font-size: 21px; font-weight: 650; margin-top: 3px;
  font-variant-numeric: tabular-nums; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 14px; }
.panel { background: #fff; border: 1px solid #e2e8f0; border-radius: 9px;
  overflow: hidden; }
.panel svg { display: block; width: 100%; height: auto; }
table { width: 100%; border-collapse: collapse; background: #fff;
  border: 1px solid #e2e8f0; border-radius: 9px; overflow: hidden;
  font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; border-bottom: 1px solid #eef2f7; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #f1f5f9; color: #475569; font-weight: 600; }
tbody tr:last-child td { border-bottom: 0; }
.notes { margin-top: 22px; color: #64748b; font-size: 12px; }
.bad { color: #b91c1c; font-weight: 600; }
.good { color: #047857; font-weight: 600; }
"""


def _phase_key(name: str) -> tuple[int, int, str]:
    match = PHASE_RE.match(name)
    if match is None:
        return (3, 9, name)
    kind, rest = match.groups()
    return (KIND_ORDER[kind], DOMAIN_ORDER.get(rest, 9), rest)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "step" in record:
                rows.append(record)
    return rows


def _phase_files(root: Path, phase: str) -> list[Path]:
    base = root / phase
    if not base.is_dir():
        return []
    files = sorted(base.glob("runs/*/metrics.jsonl"))
    if files:
        return files
    flat = base / "metrics.jsonl"
    return [flat] if flat.exists() else []


def discover_phases(root: Path) -> list[str]:
    names = [
        child.name
        for child in root.iterdir()
        if child.is_dir() and PHASE_RE.match(child.name) and _phase_files(root, child.name)
    ]
    return sorted(names, key=_phase_key)


def load_phases(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    phases = discover_phases(root)
    records: list[dict[str, Any]] = []
    offset = 0
    for phase in phases:
        match = PHASE_RE.match(phase)
        assert match is not None
        kind, domain = match.groups()
        collected: dict[int, dict[str, Any]] = {}
        for path in _phase_files(root, phase):
            for record in _read_jsonl(path):
                step = int(record["step"])
                # Later files win when a resume re-logged the boundary step.
                if step not in collected or len(record) >= len(collected[step]):
                    collected[step] = record
        ordered = [collected[step] for step in sorted(collected)]
        for record in ordered:
            row = dict(record)
            row["phase"] = phase
            row["phase_kind"] = kind
            row["phase_domain"] = domain
            row["phase_step"] = int(record["step"])
            row["global_step"] = offset + int(record["step"])
            records.append(row)
        if ordered:
            offset += int(ordered[-1]["step"]) + 1
    return records, phases


def _finite(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def _last(records: list[dict[str, Any]], key: str) -> float | None:
    for record in reversed(records):
        values = _finite([record.get(key)])
        if values:
            return values[0]
    return None


def _first(records: list[dict[str, Any]], key: str) -> float | None:
    for record in records:
        values = _finite([record.get(key)])
        if values:
            return values[0]
    return None


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = _finite([record.get(key) for record in records])
    return statistics.mean(values) if values else None


def _max(records: list[dict[str, Any]], key: str) -> float | None:
    values = _finite([record.get(key) for record in records])
    return max(values) if values else None


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def _phase_summary(phase: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    match = PHASE_RE.match(phase)
    kind = match.group(1) if match else "unknown"
    domain = match.group(2) if match else phase
    step_times = _finite([record.get("step_time_s") for record in records])
    summary: dict[str, Any] = {
        "phase": phase,
        "kind": kind,
        "domain": domain,
        "steps": len(records),
        "first_step": int(records[0]["step"]),
        "last_step": int(records[-1]["step"]),
        "wall_seconds": sum(step_times) if step_times else None,
        "mean_step_time_s": statistics.mean(step_times) if step_times else None,
        "mean_tokens_per_sec": _mean(records, "tokens_per_sec"),
        "peak_memory_mb": _max(records, "mem_peak_mb"),
        "first_loss": _first(records, "loss"),
        "last_loss": _last(records, "loss"),
        "first_main_loss": _first(records, "main_loss"),
        "last_main_loss": _last(records, "main_loss"),
        "max_tokens_seen": _max(records, "tokens_seen"),
        "response_tokens": sum(_finite([record.get("response_tokens") for record in records])),
        "first_reward_mean": _first(records, "reward_mean"),
        "last_reward_mean": _last(records, "reward_mean"),
        "mean_reward_mean": _mean(records, "reward_mean"),
        "mean_finished_rate": _mean(records, "finished_rate"),
        "mean_max_new_hit_rate": _mean(records, "max_new_hit_rate"),
        "first_kl_total": _first(records, "kl_total"),
        "last_kl_total": _last(records, "kl_total"),
        "mean_kl_total": _mean(records, "kl_total"),
    }
    for key in ("kl_general", "kl_code", "kl_tool"):
        summary[f"first_{key}"] = _first(records, key)
        summary[f"last_{key}"] = _last(records, key)
    for key in ("unit_tests", "reward_model", "unfinished", "format", "tool_alignment"):
        values = _finite([record.get(f"reward_{key}") for record in records])
        summary[f"mean_reward_{key}"] = statistics.mean(values) if values else None
    return summary


def _sft_panel(records: list[dict[str, Any]], phases: list[str]) -> list[str]:
    panels: list[str] = []
    chart = Chart("SFT main loss", xlabel="phase step", ylabel="loss")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        color = DOMAIN_COLOR.get(phase.removeprefix("sft_"), PALETTE[0])
        chart.add_series(phase, [r["phase_step"] for r in rows],
                         [r["main_loss"] for r in rows if "main_loss" in r], color=color)
    panels.append(chart.render())

    chart = Chart("SFT per-domain main loss", xlabel="phase step", ylabel="loss")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        for domain in ("general", "code", "tool"):
            key = f"train_main_loss_{domain}"
            if any(key in record for record in rows):
                chart.add_series(f"{phase.removeprefix('sft_')}/{domain}",
                                 [r["phase_step"] for r in rows],
                                 [r[key] for r in rows if key in r],
                                 color=DOMAIN_COLOR[domain],
                                 dash=None if phase.endswith(domain) else "--")
    panels.append(chart.render())

    chart = Chart("SFT step time", xlabel="phase step", ylabel="seconds")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        chart.add_series(phase, [r["phase_step"] for r in rows],
                         [r["step_time_s"] for r in rows if "step_time_s" in r])
    panels.append(chart.render())

    chart = Chart("SFT target-token throughput", xlabel="phase step", ylabel="tokens/s")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        chart.add_series(phase, [r["phase_step"] for r in rows],
                         [r["tokens_per_sec"] for r in rows if "tokens_per_sec" in r])
    panels.append(chart.render())
    return panels


def _grpo_panel(records: list[dict[str, Any]], phases: list[str]) -> list[str]:
    if not phases:
        return []
    panels: list[str] = []
    chart = Chart("GRPO reward mean", xlabel="phase step", ylabel="reward")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        domain = phase.removeprefix("grpo_")
        chart.add_series(phase, [r["phase_step"] for r in rows],
                         [r["reward_mean"] for r in rows if "reward_mean" in r],
                         color=DOMAIN_COLOR.get(domain, PALETTE[0]))
    panels.append(chart.render())

    chart = Chart("GRPO reward std", xlabel="phase step", ylabel="std")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        domain = phase.removeprefix("grpo_")
        chart.add_series(phase, [r["phase_step"] for r in rows],
                         [r["reward_std"] for r in rows if "reward_std" in r],
                         color=DOMAIN_COLOR.get(domain, PALETTE[0]))
    panels.append(chart.render())

    chart = Chart("GRPO response length", xlabel="phase step", ylabel="tokens")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        domain = phase.removeprefix("grpo_")
        chart.add_series(phase, [r["phase_step"] for r in rows],
                         [r["response_length_mean"] for r in rows
                          if "response_length_mean" in r],
                         color=DOMAIN_COLOR.get(domain, PALETTE[0]))
    panels.append(chart.render())

    chart = Chart("GRPO stop rate", xlabel="phase step", ylabel="fraction")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        domain = phase.removeprefix("grpo_")
        chart.add_series(f"{phase}/finished", [r["phase_step"] for r in rows],
                         [r["finished_rate"] for r in rows if "finished_rate" in r],
                         color=DOMAIN_COLOR.get(domain, PALETTE[0]))
        chart.add_series(f"{phase}/max_new", [r["phase_step"] for r in rows],
                         [r["max_new_hit_rate"] for r in rows if "max_new_hit_rate" in r],
                         color=DOMAIN_COLOR.get(domain, PALETTE[0]), dash="--")
    panels.append(chart.render())

    chart = Chart("GRPO loss", xlabel="phase step", ylabel="loss")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        domain = phase.removeprefix("grpo_")
        chart.add_series(phase, [r["phase_step"] for r in rows],
                         [r["loss"] for r in rows if "loss" in r],
                         color=DOMAIN_COLOR.get(domain, PALETTE[0]))
    panels.append(chart.render())

    chart = Chart("GRPO grad norm", xlabel="phase step", ylabel="|g|")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        domain = phase.removeprefix("grpo_")
        chart.add_series(phase, [r["phase_step"] for r in rows],
                         [r["grad_norm"] for r in rows if "grad_norm" in r],
                         color=DOMAIN_COLOR.get(domain, PALETTE[0]))
    panels.append(chart.render())

    chart = Chart("GRPO reward components", xlabel="phase step", ylabel="mean")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        domain = phase.removeprefix("grpo_")
        for key, dash in (("unit_tests", None), ("reward_model", "--"),
                          ("unfinished", ":"), ("format", "-.")):
            values = [r.get(f"reward_{key}") for r in rows if f"reward_{key}" in r]
            if not values:
                continue
            chart.add_series(f"{phase}/{key}", [r["phase_step"] for r in rows
                                               if f"reward_{key}" in r],
                             values, color=DOMAIN_COLOR.get(domain, PALETTE[0]),
                             dash=dash)
    panels.append(chart.render())
    return panels


def _opd_panel(records: list[dict[str, Any]], phases: list[str]) -> list[str]:
    if not phases:
        return []
    panels: list[str] = []
    chart = Chart("OPD reverse KL", xlabel="phase step", ylabel="KL")
    for key, color in (("kl_total", PALETTE[0]), ("kl_general", PALETTE[1]),
                       ("kl_code", PALETTE[2]), ("kl_tool", PALETTE[3])):
        for phase in phases:
            rows = [record for record in records if record["phase"] == phase]
            values = [r.get(key) for r in rows if key in r]
            if values:
                chart.add_series(f"{phase}/{key}", [r["phase_step"] for r in rows
                                                   if key in r], values, color=color)
    panels.append(chart.render())

    chart = Chart("OPD loss", xlabel="phase step", ylabel="loss")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        chart.add_series(phase, [r["phase_step"] for r in rows],
                         [r["loss"] for r in rows if "loss" in r])
    panels.append(chart.render())

    chart = Chart("OPD grad norm", xlabel="phase step", ylabel="|g|")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        chart.add_series(phase, [r["phase_step"] for r in rows],
                         [r["grad_norm"] for r in rows if "grad_norm" in r])
    panels.append(chart.render())

    chart = Chart("OPD step time", xlabel="phase step", ylabel="seconds")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        chart.add_series(phase, [r["phase_step"] for r in rows],
                         [r["step_time_s"] for r in rows if "step_time_s" in r])
    panels.append(chart.render())
    return panels


def _global_panels(records: list[dict[str, Any]], phases: list[str]) -> list[str]:
    panels: list[str] = []
    chart = Chart("All phases: step time", xlabel="global step", ylabel="seconds")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        if rows:
            panel = [r for r in rows if "step_time_s" in r]
            chart.add_series(phase, [r["global_step"] for r in panel],
                             [r["step_time_s"] for r in panel])
    panels.append(chart.render())

    chart = Chart("All phases: throughput", xlabel="global step", ylabel="tokens/s")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        if rows:
            chart.add_series(phase, [r["global_step"] for r in rows if "tokens_per_sec" in r],
                             [r["tokens_per_sec"] for r in rows if "tokens_per_sec" in r])
    panels.append(chart.render())

    chart = Chart("All phases: peak memory", xlabel="global step", ylabel="MiB")
    for phase in phases:
        rows = [record for record in records if record["phase"] == phase]
        if rows:
            chart.add_series(phase, [r["global_step"] for r in rows if "mem_peak_mb" in r],
                             [r["mem_peak_mb"] for r in rows if "mem_peak_mb" in r])
    panels.append(chart.render())
    return panels


def _cards(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> str:
    total_steps = sum(item["steps"] for item in summary)
    wall = sum(item["wall_seconds"] or 0.0 for item in summary)
    peak = max((item["peak_memory_mb"] or 0.0 for item in summary), default=0.0)
    sft = [item for item in summary if item["kind"] == "sft"]
    grpo = [item for item in summary if item["kind"] == "grpo"]
    opd = [item for item in summary if item["kind"] == "opd"]
    cards = [
        ("phases", str(len(summary))),
        ("optimizer steps", f"{total_steps}"),
        ("wall time", f"{wall / 60:.1f} min"),
        ("peak memory", f"{peak / 1024:.1f} GiB"),
    ]
    if sft:
        cards.append((
            "SFT last loss",
            " / ".join(f"{item['domain']} {_fmt(item['last_loss'], 2)}" for item in sft),
        ))
    if grpo:
        cards.append((
            "GRPO last reward",
            " / ".join(
                f"{item['domain']} {_fmt(item['last_reward_mean'], 2)}" for item in grpo
            ),
        ))
    if opd:
        kl = opd[-1]
        cards.append(("OPD KL first → last",
                      f"{_fmt(kl['first_kl_total'])} → {_fmt(kl['last_kl_total'])}"))
    return "".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )


def _phase_table(summary: list[dict[str, Any]]) -> str:
    head = ("phase", "steps", "wall", "s/step", "tok/s", "peak GiB",
            "first loss", "last loss", "reward", "KL")
    rows = []
    for item in summary:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['phase'])}</td>"
            f"<td>{item['steps']}</td>"
            f"<td>{_fmt((item['wall_seconds'] or 0) / 60, 1)}m</td>"
            f"<td>{_fmt(item['mean_step_time_s'], 2)}</td>"
            f"<td>{_fmt(item['mean_tokens_per_sec'], 0)}</td>"
            f"<td>{_fmt((item['peak_memory_mb'] or 0) / 1024, 2)}</td>"
            f"<td>{_fmt(item['first_loss'])}</td>"
            f"<td>{_fmt(item['last_loss'])}</td>"
            f"<td>{_fmt(item['last_reward_mean'])}</td>"
            f"<td>{_fmt(item['last_kl_total'])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(name)}</th>" for name in head)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _coverage_table(root: Path) -> str:
    path = root / "coverage.json"
    if not path.exists():
        return ""
    try:
        coverage = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    base = coverage.get("base")
    prefix = f'<p class="sub">{html.escape(str(base))}</p>' if base else ""
    rows = []
    for item in coverage.get("phases", []):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('phase', '')))}</td>"
            f"<td>{html.escape(str(item.get('available', '—')))}</td>"
            f"<td>{html.escape(str(item.get('consumed', '—')))}</td>"
            f"<td>{html.escape(str(item.get('epochs', '—')))}</td>"
            f"<td>{html.escape(str(item.get('note', '')))}</td>"
            "</tr>"
        )
    return (
        prefix
        + "<table><thead><tr><th>phase</th><th>available pool</th>"
        "<th>consumed</th><th>epochs / coverage</th><th>note</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_report(root: Path, out: Path, *, title: str = "miniDeepSeek post-training report") -> dict[str, Any]:
    records, phases = load_phases(root)
    if not records:
        raise SystemExit(f"no sft_/grpo_/opd_ metrics under {root}")
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_phase[record["phase"]].append(record)
    summary = [_phase_summary(phase, by_phase[phase]) for phase in phases]

    sft_phases = [phase for phase in phases if phase.startswith("sft_")]
    grpo_phases = [phase for phase in phases if phase.startswith("grpo_")]
    opd_phases = [phase for phase in phases if phase.startswith("opd_")]

    panels: list[str] = []
    panels += _sft_panel(records, sft_phases)
    panels += _grpo_panel(records, grpo_phases)
    panels += _opd_panel(records, opd_phases)
    panels += _global_panels(records, phases)

    meta = {
        "root": str(root),
        "records": len(records),
        "phases": ", ".join(phases),
    }
    meta_html = "".join(
        f"<span>{html.escape(str(key))}: {html.escape(str(value))}</span>"
        for key, value in meta.items()
    )
    report = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="sub">merged from {len(records)} metrics records ·
    {len(phases)} phases · latest step {max(r['global_step'] for r in records)}</div>
  <div class="meta">{meta_html}</div>
</header>
<section class="cards">{_cards(summary, records)}</section>
<section class="grid">{''.join(f'<div class="panel">{svg}</div>' for svg in panels)}</section>
<h2>Phase summary</h2>
{_phase_table(summary)}
<h2>Data coverage</h2>
{_coverage_table(root) or '<p class="sub">未提供 coverage.json。</p>'}
<h2>Notes</h2>
<div class="notes">
  SFT 的 loss 是 response-only target loss，tok/s 是 target token 吞吐；
  GRPO/OPD 的 tok/s 是 response token 吞吐；OPD 的 KL 是 full-vocabulary
  reverse KL。每个 stage 的原始记录仍在各自 <code>metrics.jsonl</code>，本页只做合并展示。
  GRPO/OPD 本轮尚未写 <code>mem_peak_mb</code>（代码已补，下轮生效）；外部
  nvidia-smi 观测约为 GRPO general 9 GiB、GRPO tool 19.7 GiB、OPD 8.6 GiB。
</div>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    return {"path": str(out), "records": len(records), "phases": phases, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="post-training root, e.g. checkpoints/posttrain_e2e_4k_145m")
    parser.add_argument("--out", default=None, help="HTML output path")
    parser.add_argument("--title", default="miniDeepSeek post-training report")
    parser.add_argument("--write-metrics", action="store_true", default=True,
                        help="write metrics_all.jsonl and summary.json (default on)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    if not root.is_dir():
        raise SystemExit(f"no such root: {root}")
    records, phases = load_phases(root)
    if not records:
        raise SystemExit(f"no sft_/grpo_/opd_ metrics under {root}")
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_phase[record["phase"]].append(record)
    summary = [_phase_summary(phase, by_phase[phase]) for phase in phases]
    if args.write_metrics:
        metrics_path = root / "metrics_all.jsonl"
        with metrics_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary_path = root / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "root": str(root),
                    "records": len(records),
                    "phases": phases,
                    "totals": {
                        "steps": sum(item["steps"] for item in summary),
                        "wall_seconds": sum(item["wall_seconds"] or 0 for item in summary),
                        "peak_memory_mb": max(
                            (item["peak_memory_mb"] or 0 for item in summary), default=0
                        ),
                    },
                    "phase_summary": summary,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    out = Path(args.out) if args.out else root / "report.html"
    result = build_report(root, out, title=args.title)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
