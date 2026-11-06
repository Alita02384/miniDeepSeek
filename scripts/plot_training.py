#!/usr/bin/env python3
"""Render ``metrics.jsonl`` into a static HTML report with inline SVG charts.

Usage::

    python scripts/plot_training.py checkpoints/pretrain/metrics.jsonl
    python scripts/plot_training.py run/metrics.jsonl --out run/report.html \
        --meta model=mini_220m_a80m --meta run=pretrain_full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minideepseek.viz import build_report, load_metrics  # noqa: E402


def _parse_meta(values: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator:
            raise SystemExit(f"--meta expects key=value, got {value!r}")
        meta[key] = item
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", help="one or more metrics.jsonl files")
    parser.add_argument("--out", default=None, help="output HTML path")
    parser.add_argument("--title", default="miniDeepSeek training report")
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="run metadata shown in the header (repeatable)",
    )
    args = parser.parse_args(argv)

    records = []
    paths = [Path(path) for path in args.metrics]
    for path in paths:
        records.extend(load_metrics(path))
    if not records:
        raise SystemExit("no metrics records found")
    records.sort(key=lambda record: float(record.get("step", 0)))

    out = Path(args.out) if args.out else paths[0].parent / "report.html"
    meta = _parse_meta(args.meta)
    meta.setdefault("metrics", ", ".join(str(path) for path in paths))
    summary = build_report(records, out, title=args.title, meta=meta)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
