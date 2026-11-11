#!/usr/bin/env python3
"""Merge every record of one curriculum run into a single sorted file.

A 145M run leaves its history in three places, with three different shapes:

* ``<out>/<stage>/runs/<run_id>/metrics.jsonl`` -- the structured record, one
  JSON object per logged step, written by :class:`MetricsLogger`;
* ``<out>/<stage>/runs/<run_id>/metrics.jsonl`` from *earlier* launches --
  same shape, different run id, sometimes overlapping step ranges because a
  restart re-logs the boundary step;
* ``logs/<config>.log`` -- the console echo.  For the very first steps of the
  campaign this is the *only* surviving copy: the structured logger only
  started at step 90, so steps 0-80 exist nowhere else.

Reading a run by hand therefore means knowing which file covers which steps and
which of two records for the same step is the authoritative one.  This script
does that once: it walks all three sources, dedupes by step (structured beats
console, and among structured records the one with more fields wins), tags each
record with ``stage`` / ``seq_len`` / ``source``, and writes
``<out>/metrics_all.jsonl`` sorted by step.

    python scripts/merge_metrics.py checkpoints/curriculum_145m_3b            # report
    python scripts/merge_metrics.py checkpoints/curriculum_145m_3b --write    # + file

The report is the point: it prints, per stage, which steps are covered and which
records had to come from the console log, so "is anything missing" is answered
by running one command instead of by grepping six directories.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: ``stage_01_seq4096`` -> (1, 4096)
STAGE_DIR = re.compile(r"stage_(\d+)_seq(\d+)$")

#: Console log line fields.  Parsed with individual regexes rather than by
#: splitting on ``|`` because the gradient-norm field is itself ``|g|``.
LOG_STEP = re.compile(r"^step\s+(\d+)")
LOG_LOSS = re.compile(r"loss\s+([\d.]+)\s+\(ema\s+([\d.]+)\)")
LOG_MAIN = re.compile(r"\bmain\s+([\d.]+)")
LOG_MTP = re.compile(r"\bmtp\s+([\d.]+)")
LOG_LR = re.compile(r"\blr\s+([\d.eE+-]+)")
LOG_GRAD = re.compile(r"\|g\|\s+([\d.]+)")
LOG_RATE = re.compile(r"([\d.]+)k?\s+tok/s")
LOG_MEM = re.compile(r"([\d.]+)\s+GiB peak")
LOG_ETA = re.compile(r"ETA\s+([\d.]+)([smhd])")
LOG_VAL = re.compile(r"val\s+([\d.]+)\s+ppl\s+([\d.]+)")
_ETA_SCALE = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_console_line(line: str) -> dict | None:
    """One ``metrics.jsonl``-shaped record from a console line, or ``None``.

    Only the fields the console actually printed survive; a record recovered
    this way is a loss curve, not a full metric row.
    """
    step = LOG_STEP.match(line)
    if step is None:
        return None
    record: dict = {"step": int(step.group(1))}
    if match := LOG_LOSS.search(line):
        record["loss"] = float(match.group(1))
        record["loss_ema"] = float(match.group(2))
    elif match := re.search(r"\bloss\s+([\d.]+)", line):
        record["loss"] = float(match.group(1))
    if match := LOG_MAIN.search(line):
        record["main_loss"] = float(match.group(1))
    if match := LOG_MTP.search(line):
        record["mtp_loss"] = float(match.group(1))
    if match := LOG_LR.search(line):
        record["lr"] = float(match.group(1))
    if match := LOG_GRAD.search(line):
        record["grad_norm"] = float(match.group(1))
    if match := LOG_RATE.search(line):
        # "6.30k tok/s" -> 6300
        value = float(match.group(1))
        record["tokens_per_sec"] = value * 1000.0 if "k tok/s" in line else value
    if match := LOG_MEM.search(line):
        record["mem_peak_mb"] = float(match.group(1)) * 1024.0
    if match := LOG_ETA.search(line):
        record["eta_seconds"] = float(match.group(1)) * _ETA_SCALE[match.group(2)]
    if match := LOG_VAL.search(line):
        record["val_loss"] = float(match.group(1))
        record["val_ppl"] = float(match.group(2))
    return record if len(record) > 1 else None


def stage_of(path: Path) -> tuple[int | None, int | None]:
    """``(stage_index, seq_len)`` from a ``.../stage_01_seq4096/...`` path."""
    for part in reversed(path.parts):
        match = STAGE_DIR.match(part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def read_jsonl(path: Path) -> list[dict]:
    rows = []
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


def collect(out_dir: Path, logs_dir: Path) -> tuple[dict[int, dict], list[dict]]:
    """Return ``(by_step, provenance)`` for every record the run ever wrote."""
    records: dict[int, dict] = {}
    provenance: list[dict] = []

    structured: list[tuple[Path, list[dict]]] = []
    for pattern in ("*/runs/*/metrics.jsonl", "*/metrics.jsonl", "runs/*/metrics.jsonl"):
        for path in sorted(out_dir.glob(pattern)):
            rows = read_jsonl(path)
            if rows:
                structured.append((path, rows))
    for path, rows in structured:
        stage, seq_len = stage_of(path)
        provenance.append(
            {
                "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "kind": "metrics",
                "records": len(rows),
                "steps": (min(r["step"] for r in rows), max(r["step"] for r in rows)),
                "stage": stage,
                "seq_len": seq_len,
            }
        )
        for record in rows:
            step = int(record["step"])
            record = dict(record)
            record.setdefault("stage", stage)
            record.setdefault("seq_len", seq_len)
            record["source"] = "metrics"
            previous = records.get(step)
            # More fields wins: a resumed launch re-logs the boundary step, and
            # the earlier copy may predate a metric being added.
            if previous is None or len(record) >= len(previous):
                records[step] = record

    for path in sorted(logs_dir.glob("*.log")):
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                row = parse_console_line(line)
                if row is not None:
                    rows.append(row)
        if not rows:
            continue
        stage, seq_len = stage_of(path)  # logs carry no stage in their path
        provenance.append(
            {
                "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "kind": "console",
                "records": len(rows),
                "steps": (min(r["step"] for r in rows), max(r["step"] for r in rows)),
                "stage": stage,
                "seq_len": seq_len,
            }
        )
        for record in rows:
            step = int(record["step"])
            if step in records:
                continue  # structured records always outrank the console echo
            record = dict(record)
            record["stage"] = None
            record["seq_len"] = None
            record["source"] = "console"
            records[step] = record
    return records, provenance


def coverage(records: dict[int, dict]) -> dict[str, object]:
    steps = sorted(records)
    gaps = []
    run_start = None
    for step in steps:
        if run_start is None:
            run_start = step
            continue
        if step - run_start > 1:
            gaps.append((run_start + 1, step - 1))
        run_start = step
    console_only = sorted(step for step, rec in records.items() if rec.get("source") == "console")
    return {
        "steps": len(steps),
        "first": steps[0] if steps else None,
        "last": steps[-1] if steps else None,
        "gaps": gaps,
        "console_only": console_only,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="curriculum out_dir, e.g. checkpoints/curriculum_145m_3b")
    parser.add_argument("--logs-dir", default="logs", help="directory holding the console logs")
    parser.add_argument("--write", action="store_true", help="write <out_dir>/metrics_all.jsonl")
    parser.add_argument("--name", default="metrics_all.jsonl")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if not out_dir.is_dir():
        raise SystemExit(f"no such out_dir: {out_dir}")
    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_absolute():
        logs_dir = ROOT / logs_dir

    records, provenance = collect(out_dir, logs_dir)
    if not records:
        raise SystemExit(f"no records found under {out_dir} or {logs_dir}")

    print(f"=== sources ({len(provenance)}) ===")
    for item in sorted(provenance, key=lambda p: p["steps"][0]):
        stage = "-" if item["stage"] is None else f"stage {item['stage']}"
        print(
            f"  {item['kind']:7} {stage:8} steps {item['steps'][0]:>5}..{item['steps'][1]:<5}"
            f" {item['records']:>5} rows  {item['source']}"
        )

    stats = coverage(records)
    print("\n=== merged coverage ===")
    print(f"  steps {stats['first']}..{stats['last']}  ({stats['steps']} logged steps)")
    if stats["console_only"]:
        print(
            f"  recovered from the console log only ({len(stats['console_only'])}): "
            f"{stats['console_only'][:12]}{' …' if len(stats['console_only']) > 12 else ''}"
        )
    gaps = [g for g in stats["gaps"] if g[1] - g[0] > 400]
    if gaps:
        print(f"  long gaps with no record at all: {gaps}")
    else:
        print("  no gap longer than 400 steps")
    by_stage: dict[object, list[int]] = {}
    for step, record in records.items():
        by_stage.setdefault(record.get("stage"), []).append(step)
    print("  per stage:")
    for stage in sorted(by_stage, key=lambda s: (s is None, s)):
        steps = sorted(by_stage[stage])
        label = "unknown (console)" if stage is None else f"stage {stage}"
        print(f"    {label:18} steps {steps[0]:>5}..{steps[-1]:<5} ({len(steps)})")

    if not args.write:
        print("\n(report only; pass --write to emit metrics_all.jsonl)")
        return 0
    target = out_dir / args.name
    with target.open("w", encoding="utf-8") as handle:
        for step in sorted(records):
            handle.write(json.dumps(records[step], ensure_ascii=False) + "\n")
    print(f"\nwrote {target}  ({target.stat().st_size / 1024:.0f} KB, {len(records)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
