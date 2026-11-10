#!/usr/bin/env python3
"""Render metrics into a static HTML report with inline SVG charts.

Every launch writes its own ``<out_dir>/runs/<run_id>/metrics.jsonl``, so this
script accepts any of:

    python scripts/plot_training.py checkpoints/<stage>/                     # new layout
    python scripts/plot_training.py checkpoints/<stage>/runs/<run_id>/      # one launch
    python scripts/plot_training.py checkpoints/<stage>/runs/<id>/metrics.jsonl
    python scripts/plot_training.py run/metrics.jsonl                       # legacy flat file

``--run`` decides what a stage directory expands to: ``lineage`` (default, the
latest launch plus every run it resumed from), ``latest`` (just the newest
launch), ``all`` (every launch, in launch order) or a run-id prefix.

Usage::

    python scripts/plot_training.py checkpoints/curriculum_145m_3b/stage_00_seq2048
    python scripts/plot_training.py checkpoints/.../stage_00_seq2048 --run all
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

from minideepseek import runs  # noqa: E402
from minideepseek.viz import build_report, load_metrics  # noqa: E402

SELECT_CHOICES = ("lineage", "latest", "all")


def _parse_meta(values: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator:
            raise SystemExit(f"--meta expects key=value, got {value!r}")
        meta[key] = item
    return meta


def _stage_dir(path: Path) -> Path | None:
    """The stage directory if ``path`` holds (or is) a ``runs/`` tree."""
    if path.is_dir() and (path / runs.RUNS_DIRNAME).is_dir():
        return path
    if path.is_dir() and path.name == runs.RUNS_DIRNAME:
        return path.parent
    return None


def _resolve_stage(stage: Path, select: str) -> tuple[list[dict], dict[str, str]]:
    infos = runs.list_runs(stage)
    if not infos:
        legacy = runs.legacy_metrics_path(stage)
        if legacy is None:
            raise SystemExit(f"no runs/ and no metrics.jsonl under {stage}")
        print(f"[plot_training] legacy flat metrics: {legacy}", file=sys.stderr)
        return load_metrics(legacy), {"metrics": str(legacy)}

    if select == "all":
        chosen = infos
    elif select in ("lineage", "latest"):
        chain = runs.lineage(stage)
        if not chain:
            raise SystemExit(f"no run under {stage} has any logged step yet")
        chosen = chain if select == "lineage" else [chain[-1]]
    else:
        found = runs.find_run(stage, select)
        if found is None:
            available = ", ".join(info.run_id for info in infos)
            raise SystemExit(f"no run matching {select!r}; available: {available}")
        chosen = [found]

    records = runs.collect_records(chosen)
    meta = {
        "runs": len(chosen),
        "run": chosen[-1].run_id,
        "steps": f"{min(r['step'] for r in records)}–{max(r['step'] for r in records)}",
    }
    if len(chosen) > 1:
        meta["run_ids"] = ", ".join(info.path.name.split("_")[0] for info in chosen)
    return records, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "metrics",
        help="metrics.jsonl, a run directory, or a stage directory containing runs/",
    )
    parser.add_argument("--out", default=None, help="output HTML path")
    parser.add_argument("--title", default="miniDeepSeek training report")
    parser.add_argument(
        "--run",
        default="lineage",
        help="for a stage directory: lineage (default) | latest | all | <run-id prefix>",
    )
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="run metadata shown in the header (repeatable)",
    )
    args = parser.parse_args(argv)

    path = Path(args.metrics)
    meta = _parse_meta(args.meta)
    default_out = path / "report.html" if path.is_dir() else path.parent / "report.html"

    if path.is_file():
        records = load_metrics(path)
        meta.setdefault("metrics", str(path))
    elif (stage := _stage_dir(path)) is not None:
        records, run_meta = _resolve_stage(stage, args.run)
        meta = {**run_meta, **meta}
        default_out = stage / "report.html"
    elif (path / runs.METRICS_FILENAME).exists():
        # A single run directory: render exactly that launch.
        records = load_metrics(path / runs.METRICS_FILENAME)
        meta = {"run": path.name, **meta}
    else:
        raise SystemExit(f"{path} is neither a metrics file, a run dir, nor a stage dir")

    if not records:
        raise SystemExit("no metrics records found")
    records.sort(key=lambda record: float(record.get("step", 0)))

    out = Path(args.out) if args.out else default_out
    summary = build_report(records, out, title=args.title, meta=meta)
    summary["runs"] = meta.get("runs")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
