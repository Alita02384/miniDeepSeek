#!/usr/bin/env python3
"""Fit and plot the pretraining scaling law from validation loss records.

Reads one or more ``metrics.jsonl`` files, reconstructs ``(tokens, val_loss)``
points, fits ``L = c · T^α`` in log-log space, and writes an HTML report with
the data, the fit line, and an optional Chinchilla reference marker.

Usage::

    python scripts/scaling_law.py run/metrics.jsonl --params-m 216.65 \
        --out run/scaling_law.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minideepseek.train.scaling import (  # noqa: E402
    PowerLawFit,
    ValidationPoint,
    fit_power_law,
    validation_points,
)
from minideepseek.viz import PALETTE, Annotation, Chart, load_metrics  # noqa: E402

_CSS = """
body { margin: 0; padding: 32px; background: #f8fafc; color: #0f172a;
       font-family: ui-sans-serif, -apple-system, "Segoe UI", sans-serif; }
h1 { margin: 0 0 6px; font-size: 24px; }
.sub { color: #64748b; font-size: 13px; margin-bottom: 18px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
         gap: 10px; margin-bottom: 16px; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; }
.card-label { color: #64748b; font-size: 11px; }
.card-value { font-size: 22px; font-weight: 650; margin-top: 4px; }
.panel { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 16px;
        background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }
th, td { padding: 7px 10px; border-bottom: 1px solid #eef2f7; text-align: right;
         font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
th { background: #f8fafc; color: #64748b; }
.note { color: #64748b; font-size: 12px; margin-top: 12px; }
"""


def _fmt(value: float, digits: int = 4) -> str:
    return "—" if not math.isfinite(value) else f"{value:.{digits}f}"


def _build_chart(points: list[ValidationPoint], fits: dict[str, PowerLawFit], chinchilla: float | None) -> str:
    chart = Chart(
        title="Validation loss vs tokens",
        xlabel="tokens",
        ylabel="val loss",
        width=760,
        height=380,
        logx=True,
        logy=True,
    )
    runs = sorted({point.run for point in points})
    for index, run in enumerate(runs):
        run_points = [point for point in points if point.run == run]
        color = PALETTE[index % len(PALETTE)]
        chart.add_series(
            run,
            [point.tokens for point in run_points],
            [point.loss for point in run_points],
            color=color,
            kind="points",
        )
        fit = fits.get(run)
        if fit is None:
            continue
        low = min(point.tokens for point in run_points)
        high = max(point.tokens for point in run_points)
        if chinchilla:
            high = max(high, chinchilla)
        tokens = np.logspace(math.log10(low), math.log10(high * 1.5), 80)
        chart.add_series(
            f"{run} fit",
            tokens.tolist(),
            [fit.predict(float(token)) for token in tokens],
            color=color,
            width=1.6,
            dash="5 3",
        )
    if chinchilla:
        chart.annotations.append(Annotation(x=chinchilla, label="Chinchilla 20× params"))
    return chart.render()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", help="metrics.jsonl file(s), one per run")
    parser.add_argument("--params-m", type=float, default=None, help="model parameters in millions")
    parser.add_argument("--min-tokens", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=float, default=None)
    parser.add_argument("--out", default=None, help="output HTML path")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--title", default="miniDeepSeek scaling law")
    args = parser.parse_args(argv)

    points: list[ValidationPoint] = []
    fits: dict[str, PowerLawFit] = {}
    for path_str in args.metrics:
        path = Path(path_str)
        run = path.parent.name or path.stem
        run_points = validation_points(
            load_metrics(path),
            run=run,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
        if not run_points:
            print(f"warning: no validation points in {path}", file=sys.stderr)
            continue
        points.extend(run_points)
        try:
            fits[run] = fit_power_law(
                run_points,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
            )
        except ValueError as error:
            print(f"warning: {run}: {error}", file=sys.stderr)
    if not points:
        raise SystemExit("no validation points found")
    if not fits:
        raise SystemExit("need at least 3 validation points to fit a power law")

    chinchilla = args.params_m * 20 * 1e6 if args.params_m else None
    chart = _build_chart(points, fits, chinchilla)

    rows = []
    for run, fit in fits.items():
        predicted = fit.predict(chinchilla) if chinchilla else float("nan")
        rows.append(
            "<tr>"
            f"<td>{html.escape(run)}</td>"
            f"<td>{fit.alpha:.4f}</td>"
            f"<td>{fit.coefficient:.4g}</td>"
            f"<td>{fit.r2:.4f}</td>"
            f"<td>{fit.points}</td>"
            f"<td>{_fmt(predicted)}</td>"
            "</tr>"
        )
    fit_table = (
        "<table><thead><tr><th>run</th><th>α</th><th>c</th><th>R²</th>"
        "<th>points</th><th>loss @ Chinchilla</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

    point_rows = []
    for point in points[-40:]:
        point_rows.append(
            "<tr>"
            f"<td>{html.escape(point.run)}</td>"
            f"<td>{point.step}</td>"
            f"<td>{point.tokens:,.0f}</td>"
            f"<td>{point.loss:.4f}</td>"
            "</tr>"
        )
    point_table = (
        "<table><thead><tr><th>run</th><th>step</th><th>tokens</th><th>val loss</th>"
        "</tr></thead><tbody>" + "".join(point_rows) + "</tbody></table>"
    )

    cards = []
    if args.params_m:
        cards.append(("params", f"{args.params_m:.1f}M"))
    if chinchilla:
        cards.append(("Chinchilla tokens", f"{chinchilla / 1e9:.2f}B"))
    for run, fit in fits.items():
        cards.append((f"alpha ({run})", f"{fit.alpha:.4f}"))
        cards.append((f"R2 ({run})", f"{fit.r2:.4f}"))
    cards.append(("points", str(len(points))))
    card_html = "".join(
        f'<div class="card"><div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(value)}</div></div>'
        for label, value in cards
    )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(args.title)}</title><style>{_CSS}</style></head>
<body>
<h1>{html.escape(args.title)}</h1>
<div class="sub">Power-law fit L = c · T<sup>α</sup> over {len(points)} validation points</div>
<div class="cards">{card_html}</div>
<div class="panel">{chart}</div>
{fit_table}
{point_table}
<div class="note">Fit uses ordinary least squares in log-log space; the dashed
line extends to the Chinchilla budget. Points are token-weighted validation
losses from the training log.</div>
</body></html>
"""
    out = Path(args.out) if args.out else Path(args.metrics[0]).parent / "scaling_law.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")

    summary = {
        "output": str(out),
        "points": len(points),
        "chinchilla_tokens": chinchilla,
        "fits": {run: fit.as_dict() for run, fit in fits.items()},
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
