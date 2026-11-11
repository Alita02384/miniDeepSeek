"""One-command pipeline: tokenize -> pretrain -> SFT -> GRPO -> OPD -> eval -> export.

The pipeline is intentionally a thin orchestrator over the existing entry
points.  It reads a YAML stage list, resolves configs into ``runs/<name>/``,
computes token budgets from the actual tokenization metadata (never a hard-coded
6B), and skips stages whose ``final/`` checkpoint already exists.

Example::

    python -m minideepseek.pipeline --config configs/pipeline/minimind_145m.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import deep_update, load_yaml
from .data.tokenize import tokenize_pretrain, tokenize_sft

ROOT = Path(__file__).resolve().parents[2]


def _meta_tokens(path: str | Path) -> int | None:
    meta = Path(path).with_suffix(".json")
    if not meta.exists():
        return None
    value = json.loads(meta.read_text(encoding="utf-8")).get("tokens")
    return int(value) if value else None


def _meta_field(path: str | Path, field: str) -> int | None:
    meta = Path(path).with_suffix(".json")
    if not meta.exists():
        return None
    value = json.loads(meta.read_text(encoding="utf-8")).get(field)
    return int(value) if value else None


_SEQ_SUFFIX = {2048: "2k", 4096: "4k", 8192: "8k"}


def _checkpoint_done(path: str | Path) -> bool:
    return (Path(path) / "final" / "meta.json").exists()


def _dataset_exists(path: str | Path) -> bool:
    """Whether a logical dataset path has either an archive or sidecars.

    ``save_flat_dataset`` keeps the config-visible ``<name>.npz`` path but
    writes ``<name>.input_ids.npy`` sidecars plus ``<name>.json``; checking
    ``Path.exists(path)`` alone rejects every dataset produced by the current
    pipeline.
    """
    path = Path(path)
    if path.exists():
        return True
    names = ("input_ids", "positions", "target_mask", "offsets")
    return path.with_suffix(".json").exists() and all(
        path.with_suffix(f".{name}.npy").exists() for name in names
    )


def _sft_step_plan(
    config_data: dict[str, Any],
    *,
    stage_name: str,
    epochs: float,
    tokenization_dir: Path,
) -> tuple[int, int, int]:
    """Return ``(steps, primary_target_tokens, per_step_primary_target)``.

    With ``data.sources`` the trainer mixes by *loss-token* weight, so the
    primary domain's per-step target count is
    ``batch*grad * (w_primary/Σw) / Σ(w_i/r_i)`` where ``r_i`` is each source's
    target/input ratio.  Computing steps from one npz while training on the
    source mixture is exactly the old bug.
    """
    data = config_data.get("data", {}) or {}
    batch = int(data.get("batch_tokens", 32768))
    grad_accum = int(config_data.get("train", {}).get("grad_accum", 8))
    sources = list(data.get("sources") or [])
    if not sources:
        path = tokenization_dir / f"sft_{stage_name}.npz"
        tokens = _meta_tokens(path)
        targets = _meta_field(path, "target_tokens")
        if not tokens or not targets:
            raise SystemExit(f"missing tokenized SFT dataset: {path}")
        per_step = max(batch * grad_accum, 1)
        steps = max(int(round(targets * epochs / per_step)), 1)
        return steps, targets, int(targets / steps)

    weights: list[float] = []
    ratios: list[float] = []
    targets: list[int] = []
    primary_index = None
    for index, source in enumerate(sources):
        path = Path(source["path"])
        tokens = _meta_tokens(path)
        target = _meta_field(path, "target_tokens")
        if not tokens or not target:
            raise SystemExit(f"missing tokenized SFT source: {path}")
        ratio = target / tokens
        weights.append(float(source.get("weight", 1.0)))
        ratios.append(ratio)
        targets.append(target)
        if source.get("domain") == stage_name or stage_name in path.stem:
            primary_index = index
    if primary_index is None:
        raise SystemExit(
            f"stage {stage_name!r} has no matching primary source in data.sources"
        )
    total_weight = sum(weights)
    effective_total = sum(weight / max(ratio, 1e-9) for weight, ratio in zip(weights, ratios, strict=True))
    loss_per_micro = max(batch, 1) * total_weight / max(effective_total, 1e-9)
    primary_loss_per_step = (
        loss_per_micro
        * grad_accum
        * weights[primary_index]
        / max(total_weight, 1e-9)
    )
    if primary_loss_per_step <= 0:
        raise SystemExit(f"invalid SFT mixing plan for {stage_name}")
    steps = max(int(round(targets[primary_index] * epochs / primary_loss_per_step)), 1)
    return steps, targets[primary_index], int(primary_loss_per_step)


def _pretrain_tokens(
    item: dict[str, Any],
    *,
    tokenization_dir: Path,
    domains: list[str],
    seq_len: int,
    legacy_layout: bool,
) -> tuple[int, list[dict[str, Any]]]:
    suffix = _SEQ_SUFFIX.get(seq_len, str(seq_len))
    sources = list(item.get("sources") or [])
    if not sources:
        if legacy_layout:
            sources = [{"path": str(tokenization_dir / f"pretrain_{suffix}.npz"), "weight": 1.0}]
        else:
            sources = [
                {
                    "path": str(tokenization_dir / f"pretrain_{domain}_{suffix}.npz"),
                    "weight": 1.0,
                    "domain": domain,
                }
                for domain in domains
            ]
    total = 0
    for source in sources:
        tokens = _meta_tokens(source["path"])
        if not tokens:
            raise SystemExit(f"missing tokenized pretrain source: {source['path']}")
        total += tokens
    return total, sources


def _run(command: list[str], *, log: Path, dry_run: bool) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print("+", " ".join(command))
        return
    with log.open("a", encoding="utf-8") as handle:
        handle.write("+ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, check=True, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)


def _python(module: str, args: list[str]) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _torchrun(nproc: int, module: str, args: list[str]) -> list[str]:
    if nproc > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
            "-m",
            module,
            *args,
        ]
    return _python(module, args)


def _script(script: str, args: list[str], nproc: int = 1) -> list[str]:
    if nproc > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
            script,
            *args,
        ]
    return [sys.executable, script, *args]


def _resolve_config(source: str | Path, overrides: dict[str, Any], target: Path) -> Path:
    data = load_yaml(source)
    data = deep_update(data, overrides)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


def _tokenize_stage(
    stage: dict[str, Any],
    *,
    data_dir: Path,
    tokenization_dir: Path,
    smoke: bool,
    dry_run: bool,
) -> None:
    max_records = 200 if smoke else stage.get("max_records")
    max_tokens = stage.get("max_tokens")
    seq_lens = [int(value) for value in stage.get("seq_lens", [2048, 4096, 8192])]
    domains = [str(value) for value in stage.get("data_domains", ["general"])]
    sft_domains = [str(value) for value in stage.get("sft_domains", ["general", "code", "tool"])]
    sft_budgets = stage.get(
        "sft_max_records", {"general": 50_000, "code": 30_000, "tool": 8_000}
    )
    if dry_run:
        print(f"+ tokenize pretrain domains={domains} seq_lens={seq_lens} -> {tokenization_dir}")
        print(f"+ tokenize sft domains={sft_domains} -> {tokenization_dir}")
        return
    for domain in domains:
        source = data_dir / f"pretrain_{domain}.jsonl"
        if not source.exists():
            raise SystemExit(f"missing pretrain source: {source}")
        tokenize_pretrain(
            source,
            tokenization_dir,
            seq_lens=seq_lens,
            max_records=max_records,
            max_tokens=max_tokens,
            dedup=bool(stage.get("dedup", False)),
            paragraph_dedup=bool(stage.get("paragraph_dedup", False)),
            quality_filter=not bool(stage.get("no_quality", False)),
            output_prefix=f"pretrain_{domain}",
        )
    for name in sft_domains:
        source = data_dir / f"sft_{name}.jsonl"
        if not source.exists():
            raise SystemExit(f"missing SFT source: {source}")
        budget = 200 if smoke else sft_budgets.get(name)
        tokenize_sft(
            source,
            tokenization_dir / f"sft_{name}.npz",
            seq_len=int(stage.get("sft_seq_len", 4096)),
            max_records=budget,
            max_tokens=max_tokens,
            require_tools=(name == "tool"),
        )


def _pretrain_stage(
    stage: dict[str, Any],
    *,
    run_dir: Path,
    nproc: int,
    tokenization_dir: Path,
    dry_run: bool,
) -> None:
    source = load_yaml(stage["config"])
    base = load_yaml(source.get("base_config", stage["config"]))
    ratios = {
        int(key): float(value)
        for key, value in stage.get(
            "token_ratios", {2048: 0.15, 4096: 0.55, 8192: 0.30}
        ).items()
    }
    seq_lens = sorted(ratios)
    domains = [str(value) for value in stage.get("data_domains", ["general"])]
    legacy_layout = not bool(stage.get("data_domains"))
    base_stages = source.get("stages") or []
    by_len = {int(item["seq_len"]): item for item in base_stages}
    # ``grad_accum`` lives in the base pretrain YAML, not at the curriculum
    # top level; the old fallback of 8 doubled the computed budget.
    grad_accum = int(base.get("train", {}).get("grad_accum", 16))
    total_tokens: dict[int, int] = {}
    sources_by_len: dict[int, list[dict[str, Any]]] = {}
    for seq in seq_lens:
        total, sources = _pretrain_tokens(
            by_len.get(seq) or {},
            tokenization_dir=tokenization_dir,
            domains=domains,
            seq_len=seq,
            legacy_layout=legacy_layout,
        )
        total_tokens[seq] = total
        sources_by_len[seq] = sources
    cumulative = 0
    for seq in seq_lens:
        item = by_len.get(seq)
        if item is None:
            raise SystemExit(f"curriculum has no stage for seq_len {seq}")
        if item.get("until_step") and item.get("sources"):
            print(f"pretrain {seq}: keeping configured until_step={item['until_step']}")
            cumulative = max(cumulative, int(item["until_step"]))
            continue
        batch = int(item.get("batch_tokens_final") or item.get("batch_tokens") or 4096)
        stage_tokens = int(total_tokens[seq] * ratios[seq])
        steps = max(stage_tokens // max(batch * grad_accum, 1), 1) + cumulative
        item["until_step"] = steps
        item.setdefault("sources", sources_by_len[seq])
        cumulative = steps
        print(
            f"pretrain {seq}: {stage_tokens / 1e6:.1f}M tokens -> "
            f"{steps} cumulative steps"
        )
    target = _resolve_config(
        stage["config"],
        {
            "stages": base_stages,
            "out_dir": stage.get("out_dir", "checkpoints/curriculum_145m_3b"),
        },
        run_dir / "configs" / "pretrain.yaml",
    )
    _run(
        _script("scripts/train_curriculum.py", ["--config", str(target)], nproc),
        log=run_dir / "logs" / "pretrain.log",
        dry_run=dry_run,
    )


def _sft_stage(
    stage: dict[str, Any],
    *,
    run_dir: Path,
    nproc: int,
    tokenization_dir: Path,
    dry_run: bool,
) -> None:
    config_data = load_yaml(stage["config"])
    data_cfg = config_data.get("data", {}) or {}
    epochs = float(stage.get("epochs", 1.0))
    planned_steps, primary_targets, per_step = _sft_step_plan(
        config_data,
        stage_name=str(stage["name"]),
        epochs=epochs,
        tokenization_dir=tokenization_dir,
    )
    # An explicit stage ``steps`` wins: the audited 280/115/95 budgets are
    # measured from the real mixer, while ``_sft_step_plan`` is an ideal-mixture
    # estimate that undercounts the packed-batch underfill by ~7%.
    steps = max(int(stage["steps"]), 1) if stage.get("steps") is not None else planned_steps
    overrides: dict[str, Any] = {
        "train": {
            "steps": steps,
            "resume": stage.get("resume", config_data.get("train", {}).get("resume")),
            "resume_data": False,
            "resume_optimizer": False,
            "resume_step": False,
            "out_dir": stage.get("out_dir", f"checkpoints/sft_{stage['name']}_145m"),
        },
    }
    # Do not replace a source mixture with a single tokenized_path: run.py
    # prefers data.sources whenever it exists.
    if not data_cfg.get("sources"):
        path = tokenization_dir / f"sft_{stage['name']}.npz"
        overrides["data"] = {"tokenized_path": str(path)}
    target = _resolve_config(
        stage["config"], overrides, run_dir / "configs" / f"sft_{stage['name']}.yaml"
    )
    print(
        f"sft {stage['name']}: primary target {primary_targets / 1e6:.1f}M tokens, "
        f"~{per_step} target/step -> {steps} steps "
        f"(~{steps * per_step / max(primary_targets, 1):.3f} target epoch)"
    )
    _run(
        _torchrun(
            nproc,
            "minideepseek.train.run",
            ["--config", str(target), "--mode", "sft"],
        ),
        log=run_dir / "logs" / f"sft_{stage['name']}.log",
        dry_run=dry_run,
    )


def _rl_stage(stage: dict[str, Any], *, run_dir: Path, nproc: int, dry_run: bool) -> None:
    method = stage["kind"]
    train_overrides: dict[str, Any] = {
        "resume": stage.get("resume"),
        "out_dir": stage.get("out_dir", f"checkpoints/{stage['id']}"),
    }
    if stage.get("steps") is not None:
        train_overrides["steps"] = max(int(stage["steps"]), 1)
    overrides: dict[str, Any] = {"data": {"path": stage["data"]}, "train": train_overrides}
    if stage.get("teachers"):
        overrides["teachers"] = stage["teachers"]
    target = _resolve_config(stage["config"], overrides, run_dir / "configs" / f"{stage['id']}.yaml")
    _run(_torchrun(nproc, "minideepseek.rl.run", [method, "--config", str(target)]), log=run_dir / "logs" / f"{stage['id']}.log", dry_run=dry_run)


def _eval_stage(stage: dict[str, Any], *, run_dir: Path, dry_run: bool) -> None:
    overrides = {
        "checkpoint": stage["checkpoint"],
        "data_path": stage["data"],
        "out_path": stage.get("out_path", f"reports/{stage['id']}.json"),
    }
    target = _resolve_config(stage["config"], overrides, run_dir / "configs" / f"{stage['id']}.yaml")
    _run([sys.executable, "-m", "minideepseek.eval.run", "--config", str(target)], log=run_dir / "logs" / f"{stage['id']}.log", dry_run=dry_run)


def _export_stage(stage: dict[str, Any], *, run_dir: Path, dry_run: bool) -> None:
    command = [
        sys.executable,
        "scripts/export_hf.py",
        "--checkpoint",
        stage["checkpoint"],
        "--out",
        stage.get("out", "hf/minimind_145m"),
    ]
    _run(command, log=run_dir / "logs" / "export.log", dry_run=dry_run)




def _report_stage(
    stage: dict[str, Any], *, run_dir: Path, dry_run: bool
) -> None:
    root = str(stage.get("root", stage.get("out_dir", "")))
    if not root:
        raise SystemExit(f"report stage {stage.get('id')!r} needs root/out_dir")
    out = str(stage.get("out", f"reports/{stage['id']}.html"))
    _run(
        [sys.executable, "scripts/posttraining_report.py", root, "--out", out],
        log=run_dir / "logs" / f"{stage['id']}.log",
        dry_run=dry_run,
    )


def _stage_done(stage: dict[str, Any], *, tokenization_dir: Path) -> bool:
    kind = stage["kind"]
    if kind == "tokenize":
        needed = [tokenization_dir / f"pretrain_{suffix}.json" for suffix in ("2k", "4k", "8k")]
        needed += [
            tokenization_dir / f"sft_{name}.json"
            for name in stage.get("sft_domains", ["general", "tool"])
        ]
        return all(path.exists() for path in needed)
    if kind == "curriculum":
        out = Path(stage.get("out_dir", "checkpoints/curriculum_145m_3b"))
        candidates = [out / "final" / "meta.json"]
        candidates += sorted(out.glob("stage_*")) 
        return any(
            (candidate / "final" / "meta.json").exists()
            if candidate.is_dir()
            else candidate.exists()
            for candidate in candidates
        )
    if kind in {"sft", "grpo", "opd"}:
        return (Path(stage.get("out_dir", f"checkpoints/{stage['id']}")) / "final" / "meta.json").exists()
    if kind == "eval":
        return Path(stage.get("out_path", f"reports/{stage['id']}.json")).exists()
    if kind == "report":
        return Path(stage.get("out", f"reports/{stage['id']}.html")).exists()
    if kind == "export":
        return (Path(stage.get("out", "hf/minimind_145m")) / "config.json").exists()
    return False



def _preflight(
    spec: dict[str, Any],
    data_dir: Path,
    *,
    dry_run: bool,
    start: str | None = None,
    stop: str | None = None,
) -> None:
    """Fail before a long stage if any input/resume/checkpoint is missing.

    Checkpoints produced by an earlier stage in the same run are allowed to be
    missing: a fresh SFT -> GRPO -> OPD chain necessarily has GRPO resumes and
    OPD teachers that do not exist yet.  ``start``/``stop`` limit the check to
    the stages that will actually run.
    """
    tokenization_dir = Path(spec.get("tokenization_dir", data_dir / "tokenization"))
    missing: list[str] = []
    outputs: list[Path] = []
    active = start is None

    def need(path: str | Path, *, dataset: bool = False) -> None:
        target = Path(path)
        found = _dataset_exists(target) if dataset else target.exists()
        if found:
            return
        # A path under an earlier stage's ``out_dir/final`` is this run's own
        # output, not a pre-existing input.
        if any(target == output or output in target.parents for output in outputs):
            return
        missing.append(str(target))

    for stage in spec.get("stages", []):
        stage_id = str(stage["id"])
        if start is not None and stage_id == start:
            active = True
        if not active:
            continue
        if stop is not None and stage_id == stop:
            active = False
        kind = stage["kind"]
        if kind == "tokenize":
            for domain in stage.get("data_domains", ["general"]):
                need(data_dir / f"pretrain_{domain}.jsonl")
            for name in stage.get("sft_domains", ["general", "code", "tool"]):
                need(data_dir / f"sft_{name}.jsonl")
        elif kind == "curriculum":
            need(stage["config"])
            source = load_yaml(stage["config"])
            for item in source.get("stages", []):
                for entry in item.get("sources", []):
                    need(entry["path"], dataset=True)
                for entry in item.get("val_sources", []):
                    need(entry["path"], dataset=True)
        elif kind == "sft":
            need(stage["config"])
            config_data = load_yaml(stage["config"])
            data_cfg = config_data.get("data", {}) or {}
            if data_cfg.get("sources"):
                for entry in data_cfg["sources"]:
                    need(entry["path"], dataset=True)
            else:
                need(tokenization_dir / f"sft_{stage['name']}.npz", dataset=True)
            if stage.get("resume"):
                need(stage["resume"])
        elif kind in {"grpo", "opd"}:
            need(stage["config"])
            if stage.get("data"):
                need(stage["data"])
            if stage.get("resume"):
                need(stage["resume"])
            if kind == "opd":
                # ``_rl_stage`` can replace the teacher list from the stage;
                # preflight must check the same list it will actually load.
                teachers = stage.get("teachers")
                if teachers is None:
                    teachers = load_yaml(stage["config"]).get("teachers", [])
                for teacher in teachers:
                    need(teacher["checkpoint"])
        elif kind == "eval":
            need(stage["config"])
            if stage.get("data"):
                need(stage["data"])
            if stage.get("checkpoint"):
                need(stage["checkpoint"])
        elif kind == "report":
            # The report root is this run's own output tree; preflight runs
            # before that tree exists, so only the script is checked.
            need("scripts/posttraining_report.py")
        if kind in {"sft", "grpo", "opd"}:
            outputs.append(
                Path(stage.get("out_dir", f"checkpoints/{stage_id}")) / "final"
            )
    if not missing:
        return
    message = "pipeline preflight missing paths:\n" + "\n".join(
        f"  - {path}" for path in dict.fromkeys(missing)
    )
    if dry_run:
        print(message)
    raise SystemExit(message)


def run_pipeline(config_path: str | Path, *, dry_run: bool = False, start: str | None = None, stop: str | None = None, force: bool = False) -> dict[str, Any]:
    spec = load_yaml(config_path)
    name = str(spec.get("name", "run"))
    run_dir = Path(spec.get("run_dir", f"runs/{name}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(spec.get("data_dir", "data"))
    tokenization_dir = Path(spec.get("tokenization_dir", data_dir / "tokenization"))
    nproc = int(os.environ.get("NPROC_PER_NODE", spec.get("nproc", 1)))
    _preflight(spec, data_dir, dry_run=dry_run, start=start, stop=stop)
    # A dry run must not overwrite the real state file; otherwise a later
    # look at runs/<name>/state.json reports every stage as "dry-run".
    state_path = run_dir / ("state_dry_run.json" if dry_run else "state.json")
    state: dict[str, Any] = {"name": name, "stages": {}}
    active = start is None
    for stage in spec.get("stages", []):
        stage_id = str(stage["id"])
        if start is not None and stage_id == start:
            active = True
        if not active:
            continue
        if stop is not None and stage_id == stop:
            active = False
        kind = stage["kind"]
        if not force and _stage_done(stage, tokenization_dir=tokenization_dir):
            print(f"\n== {stage_id} ({kind}) == already done, skipping")
            state["stages"][stage_id] = "skipped"
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            continue
        print(f"\n== {stage_id} ({kind}) ==")
        if kind == "tokenize":
            _tokenize_stage(stage, data_dir=data_dir, tokenization_dir=tokenization_dir, smoke=bool(spec.get("smoke")), dry_run=dry_run)
        elif kind == "curriculum":
            _pretrain_stage(stage, run_dir=run_dir, nproc=nproc, tokenization_dir=tokenization_dir, dry_run=dry_run)
        elif kind == "sft":
            _sft_stage(stage, run_dir=run_dir, nproc=nproc, tokenization_dir=tokenization_dir, dry_run=dry_run)
        elif kind in {"grpo", "opd"}:
            _rl_stage(stage, run_dir=run_dir, nproc=nproc, dry_run=dry_run)
        elif kind == "eval":
            _eval_stage(stage, run_dir=run_dir, dry_run=dry_run)
        elif kind == "report":
            _report_stage(stage, run_dir=run_dir, dry_run=dry_run)
        elif kind == "export":
            _export_stage(stage, run_dir=run_dir, dry_run=dry_run)
        else:
            raise SystemExit(f"unknown stage kind {kind!r}")
        state["stages"][stage_id] = "dry-run" if dry_run else "done"
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from", dest="start", default=None)
    parser.add_argument("--to", dest="stop", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    state = run_pipeline(args.config, dry_run=args.dry_run, start=args.start, stop=args.stop, force=args.force)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
