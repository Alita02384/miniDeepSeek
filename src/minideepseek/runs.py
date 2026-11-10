"""Run identity: one directory per training launch.

Every launch writes ``<out_dir>/runs/<run_id>/`` holding its own
``metrics.jsonl``, the resolved ``config.yaml`` and a ``meta.json`` provenance
record.  Keeping launches apart is what makes "which run produced this point"
answerable at all.

Appending every launch to a single ``<out_dir>/metrics.jsonl`` instead makes
restarts indistinguishable: a crashed run that never checkpointed restarts at
step 0, so the file ends up holding ``0,10,20,30`` twice and the loss curve of
the report zig-zags between unrelated trainings.

Resuming is still supported: :func:`lineage` walks back from a run to the run it
resumed from (by step range, newest first), so a report can span a resumed chain
without also dragging in the abandoned branches.

Deliberately torch-free: ``scripts/plot_training.py`` imports this module to
resolve paths without pulling in the training stack.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Set by a launcher so the shell log name and the run directory share one
#: timestamp even though two different processes create them.
RUN_ID_ENV = "MINIDEEPSEEK_RUN_ID"

RUNS_DIRNAME = "runs"
METRICS_FILENAME = "metrics.jsonl"
META_FILENAME = "meta.json"
CONFIG_FILENAME = "config.yaml"


def utc_stamp(moment: datetime | None = None) -> str:
    """Compact UTC timestamp, e.g. ``20261110T121245Z``."""
    moment = moment or datetime.now(UTC)
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def is_stamp(stamp: str) -> bool:
    return bool(re.fullmatch(r"\d{8}T\d{6}Z", str(stamp)))


def format_params(count: float) -> str:
    """``145.8M`` / ``1.31B`` -- short enough to sit in a directory name."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    return f"{count / 1e6:.1f}M"


def make_run_id(
    flavor: str,
    total_params: float,
    world_size: int,
    *,
    stamp: str | None = None,
) -> str:
    """``<utc stamp>_<flavor>_<params>_<world>g``, e.g. ``20261110T121245Z_mini_145m_a60m_145.8M_1g``."""
    stamp = stamp or os.environ.get(RUN_ID_ENV) or utc_stamp()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(flavor)).strip("-") or "model"
    return f"{stamp}_{slug}_{format_params(float(total_params))}_{int(world_size)}g"


def runs_root(out_dir: str | Path) -> Path:
    return Path(out_dir) / RUNS_DIRNAME


def allocate_run_dir(out_dir: str | Path, run_id: str) -> Path:
    """Create ``<out_dir>/runs/<run_id>/``.

    A second launch inside the same UTC second would otherwise share (and
    silently append to) the first one's directory, so collisions get a numeric
    suffix instead.
    """
    root = runs_root(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / run_id
    attempt = 2
    while candidate.exists():
        candidate = root / f"{run_id}-{attempt}"
        attempt += 1
    candidate.mkdir(parents=True)
    return candidate


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSONL metrics file, skipping blank lines."""
    records: list[dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return records
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def read_meta(path: str | Path) -> dict[str, Any]:
    meta_path = Path(path) / META_FILENAME
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_meta(path: str | Path, meta: dict[str, Any]) -> Path:
    target = Path(path) / META_FILENAME
    target.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return target


def update_meta(path: str | Path, **fields: Any) -> dict[str, Any]:
    """Merge ``fields`` into the run's ``meta.json`` (no-op if it is missing)."""
    meta = read_meta(path)
    if not meta:
        return meta
    meta.update(fields)
    write_meta(path, meta)
    return meta


CURRENT_FILENAME = "current.json"


def current_path(out_dir: str | Path) -> Path:
    """Where ``runs/`` records which run is currently live for ``out_dir``."""
    return runs_root(out_dir) / CURRENT_FILENAME


def read_current(out_dir: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(current_path(out_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_current(
    out_dir: str | Path, *, run_id: str, flavor: str, total_params: float
) -> Path:
    target = current_path(out_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": str(run_id),
        "flavor": str(flavor),
        "total_params": float(total_params),
        "stamp": str(run_id).split("_", 1)[0],
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def has_checkpoints(out_dir: str | Path) -> bool:
    """True when ``out_dir`` holds something a trainer could resume from."""
    root = Path(out_dir)
    if not root.is_dir():
        return False
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("step_") or child.name == "final":
            return True
    return False


def has_campaign_state(out_dir: str | Path) -> bool:
    """True when any stage under ``out_dir`` has a checkpoint."""
    if has_checkpoints(out_dir):
        return True
    root = Path(out_dir)
    if not root.is_dir():
        return False
    return any(child.is_dir() and has_checkpoints(child) for child in root.iterdir())


def continuation(out_dir: str | Path, flavor: str) -> str | None:
    """Run id this launch should append to, or ``None`` for a fresh run.

    A continuation requires both a checkpoint on disk and a live pointer that
    names the same model flavour; anything else (new model, cleaned output dir)
    starts a new run directory instead of polluting the previous one.
    """
    pointer = read_current(out_dir)
    if not pointer or not has_checkpoints(out_dir):
        return None
    if str(pointer.get("flavor") or "") != str(flavor):
        return None
    run_id = pointer.get("run_id")
    return str(run_id) if run_id else None


def record_launch(
    run_dir: str | Path,
    *,
    world_size: int,
    distributed: bool,
    distributed_mode: str,
) -> dict[str, Any]:
    """Append one launch to a run's ``meta.json`` history."""
    meta = read_meta(run_dir)
    if not meta:
        return meta
    launches = list(meta.get("launches") or [])
    launches.append(
        {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "world_size": int(world_size),
            "distributed": bool(distributed),
            "distributed_mode": str(distributed_mode),
        }
    )
    meta["launches"] = launches
    meta["last_launch_at"] = launches[-1]["at"]
    write_meta(run_dir, meta)
    return meta


def campaign_stamp(out_dir: str | Path, logs_dir: str | Path, stem: str) -> str:
    """Stamp shared by every launch of one training campaign.

    ``logs/.<config>.stamp`` remembers the first launch's stamp, so a restart
    appends to the same log instead of spawning one file per attempt.  Once
    nothing is resumable under ``out_dir`` the next launch is a fresh training
    and gets a new stamp (a new log).
    """
    logs = Path(logs_dir)
    marker = logs / f".{stem}.stamp"
    if marker.exists() and has_campaign_state(out_dir):
        recorded = marker.read_text(encoding="utf-8").strip()
        if is_stamp(recorded):
            return recorded
    stamp = utc_stamp()
    logs.mkdir(parents=True, exist_ok=True)
    marker.write_text(stamp + "\n", encoding="utf-8")
    return stamp


def launch_log_path(config_path: str | Path, logs_dir: str | Path = "logs") -> Path:
    """``logs/<config stem>_<campaign stamp>.log`` -- timestamp last."""
    import yaml  # only the launcher needs this, so keep it out of the import path

    config_path = Path(config_path)
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        config = {}
    out_dir = (
        config.get("out_dir")
        or (config.get("train") or {}).get("out_dir")
        or "checkpoints/run"
    )
    stamp = campaign_stamp(out_dir, logs_dir, config_path.stem)
    return Path(logs_dir) / f"{config_path.stem}_{stamp}.log"


def main(argv: list[str] | None = None) -> int:
    """Tiny CLI: the launcher asks for the log path it should append to."""
    import argparse

    parser = argparse.ArgumentParser(prog="minideepseek.runs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    log_command = sub.add_parser("logpath", help="print the log file for a config")
    log_command.add_argument("config")
    log_command.add_argument("--logs-dir", default="logs")
    args = parser.parse_args(argv)
    if args.command == "logpath":
        print(launch_log_path(args.config, args.logs_dir))
        return 0
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised by the launcher
    raise SystemExit(main())


@dataclass
class RunInfo:
    """One launch: where it lives and which steps it covered."""

    run_id: str
    path: Path
    started_at: str = ""
    start_step: int | None = None
    end_step: int | None = None
    records: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_records(self) -> bool:
        return self.start_step is not None and self.end_step is not None

    @property
    def label(self) -> str:
        if self.started_at:
            return f"{self.run_id} ({self.started_at})"
        return self.run_id


def summarize_run(path: str | Path) -> RunInfo:
    """Step range of one run directory, read from its metrics file."""
    run_path = Path(path)
    meta = read_meta(run_path)
    records = load_records(run_path / METRICS_FILENAME)
    steps = [int(record["step"]) for record in records if "step" in record]
    return RunInfo(
        run_id=str(meta.get("run_id") or run_path.name),
        path=run_path,
        started_at=str(meta.get("started_at") or ""),
        start_step=min(steps) if steps else None,
        end_step=max(steps) if steps else None,
        records=len(records),
        meta=meta,
    )


def list_runs(out_dir: str | Path) -> list[RunInfo]:
    """Every run directory under ``out_dir``, oldest first."""
    root = runs_root(out_dir)
    if not root.is_dir():
        return []
    infos = [summarize_run(child) for child in sorted(root.iterdir()) if child.is_dir()]
    infos.sort(key=lambda info: (info.started_at or "", info.run_id))
    return infos


def latest_run(out_dir: str | Path) -> RunInfo | None:
    infos = [info for info in list_runs(out_dir) if info.has_records]
    return infos[-1] if infos else None


def find_run(out_dir: str | Path, run_id: str) -> RunInfo | None:
    """Exact id, else unique prefix match."""
    infos = list_runs(out_dir)
    for info in infos:
        if info.run_id == run_id:
            return info
    matches = [info for info in infos if info.run_id.startswith(run_id)]
    return matches[-1] if len(matches) == 1 else None


def lineage(out_dir: str | Path, upto: str | None = None, *, at_most: int = 64) -> list[RunInfo]:
    """The run ``upto`` plus every run it resumed from, oldest first.

    A resume continues from the newest checkpoint, so the ancestor is the newest
    run whose last step is at or before this run's first step.  Runs that were
    abandoned (e.g. a crashed launch that restarted at step 0 instead of
    resuming) never match and stay out of the chain.
    """
    infos = [info for info in list_runs(out_dir) if info.has_records]
    if not infos:
        return []
    current = find_run(out_dir, upto) if upto else infos[-1]
    if current is None or not current.has_records:
        current = infos[-1]

    chain = [current]
    seen = {current.run_id}
    cursor = current.start_step
    while len(chain) < at_most:
        candidates = [
            info
            for info in infos
            if info.run_id not in seen and info.end_step is not None and info.end_step <= cursor
        ]
        if not candidates:
            break
        # Newest ancestor wins: that is the one whose checkpoint was loaded.
        previous = max(candidates, key=lambda info: (info.end_step or 0, info.started_at or ""))
        chain.insert(0, previous)
        seen.add(previous.run_id)
        cursor = previous.start_step
    return chain


def collect_records(
    run_infos: list[RunInfo], filename: str = METRICS_FILENAME
) -> list[dict[str, Any]]:
    """Concatenate the metrics of several runs, in the order given."""
    records: list[dict[str, Any]] = []
    for info in run_infos:
        records.extend(load_records(info.path / filename))
    return records


def legacy_metrics_path(out_dir: str | Path) -> Path | None:
    """The pre-run-directory flat ``metrics.jsonl``, if the stage still has one."""
    candidate = Path(out_dir) / METRICS_FILENAME
    return candidate if candidate.exists() else None
