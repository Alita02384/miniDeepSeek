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


def _checkpoint_done(path: str | Path) -> bool:
    return (Path(path) / "final" / "meta.json").exists()


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


def _tokenize_stage(stage: dict[str, Any], *, data_dir: Path, tokenization_dir: Path, smoke: bool, dry_run: bool) -> None:
    max_records = 200 if smoke else stage.get("max_records")
    max_tokens = stage.get("max_tokens")
    if dry_run:
        print(f"+ tokenize pretrain {stage.get('seq_lens')} -> {tokenization_dir}")
        print(f"+ tokenize sft general/tool -> {tokenization_dir}")
        return
    tokenize_pretrain(
        data_dir / "pretrain_t2t.jsonl",
        tokenization_dir,
        seq_lens=[int(value) for value in stage.get("seq_lens", [2048, 4096, 8192])],
        max_records=max_records,
        max_tokens=max_tokens,
        dedup=bool(stage.get("dedup", False)),
        paragraph_dedup=bool(stage.get("paragraph_dedup", False)),
        quality_filter=not bool(stage.get("no_quality", False)),
    )
    for name in ("general", "tool"):
        tokenize_sft(
            data_dir / f"sft_{name}.jsonl",
            tokenization_dir / f"sft_{name}.npz",
            seq_len=int(stage.get("sft_seq_len", 4096)),
            max_records=max_records,
            max_tokens=max_tokens,
        )


def _pretrain_stage(stage: dict[str, Any], *, run_dir: Path, nproc: int, tokenization_dir: Path, dry_run: bool) -> None:
    source = load_yaml(stage["config"])
    ratios = {int(key): float(value) for key, value in stage.get("token_ratios", {2048: 0.15, 4096: 0.55, 8192: 0.30}).items()}
    seq_lens = sorted(ratios)
    metas = {seq: _meta_tokens(tokenization_dir / f"pretrain_{ {2048: '2k', 4096: '4k', 8192: '8k'}.get(seq, str(seq)) }.npz") for seq in seq_lens}
    missing = [seq for seq, tokens in metas.items() if not tokens]
    if missing:
        raise SystemExit(f"missing tokenized pretrain datasets for seq_lens {missing}")
    total = min(int(tokens) for tokens in metas.values())
    base_stages = source.get("stages") or []
    by_len = {int(item["seq_len"]): item for item in base_stages}
    cumulative = 0
    for seq in seq_lens:
        item = by_len.get(seq)
        if item is None:
            raise SystemExit(f"curriculum has no stage for seq_len {seq}")
        batch = int(item.get("batch_tokens_final") or item.get("batch_tokens") or 4096)
        grad_accum = int(source.get("train", {}).get("grad_accum", 8))
        stage_tokens = int(total * ratios[seq])
        steps = max(stage_tokens // max(batch * grad_accum, 1), 1)
        cumulative += steps
        item["until_step"] = cumulative
        item.setdefault("tokenized_path", str(tokenization_dir / f"pretrain_{ {2048: '2k', 4096: '4k', 8192: '8k'}[seq] }.npz"))
        print(f"pretrain {seq}: {stage_tokens / 1e6:.1f}M tokens -> {steps} steps (cumulative {cumulative})")
    target = _resolve_config(
        stage["config"],
        {"stages": base_stages, "out_dir": stage.get("out_dir", "checkpoints/curriculum_145m")},
        run_dir / "configs" / "pretrain.yaml",
    )
    _run(_script("scripts/train_curriculum.py", ["--config", str(target)], nproc), log=run_dir / "logs" / "pretrain.log", dry_run=dry_run)


def _sft_stage(stage: dict[str, Any], *, run_dir: Path, nproc: int, tokenization_dir: Path, dry_run: bool) -> None:
    config_data = load_yaml(stage["config"])
    path = tokenization_dir / f"sft_{stage['name']}.npz"
    tokens = _meta_tokens(path)
    if not tokens:
        raise SystemExit(f"missing tokenized SFT dataset: {path}")
    batch = int(config_data.get("data", {}).get("batch_tokens", 32768))
    grad_accum = int(config_data.get("train", {}).get("grad_accum", 8))
    epochs = float(stage.get("epochs", 1.0))
    steps = max(int(tokens * epochs / max(batch * grad_accum, 1)), 1)
    overrides = {
        "data": {"tokenized_path": str(path)},
        "train": {
            "steps": steps,
            "resume": stage.get("resume", config_data.get("train", {}).get("resume")),
            "resume_data": False,
            "resume_optimizer": False,
            "resume_step": False,
            "out_dir": stage.get("out_dir", f"checkpoints/sft_{stage['name']}"),
        },
    }
    target = _resolve_config(stage["config"], overrides, run_dir / "configs" / f"sft_{stage['name']}.yaml")
    print(f"sft {stage['name']}: {tokens / 1e6:.1f}M tokens x {epochs} -> {steps} steps")
    _run(_torchrun(nproc, "minideepseek.train.run", ["--config", str(target), "--mode", "sft"]), log=run_dir / "logs" / f"sft_{stage['name']}.log", dry_run=dry_run)


def _rl_stage(stage: dict[str, Any], *, run_dir: Path, nproc: int, dry_run: bool) -> None:
    method = stage["kind"]
    overrides = {
        "data": {"path": stage["data"]},
        "train": {
            "resume": stage.get("resume"),
            "out_dir": stage.get("out_dir", f"checkpoints/{stage['id']}"),
        },
    }
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




def _stage_done(stage: dict[str, Any], *, tokenization_dir: Path) -> bool:
    kind = stage["kind"]
    if kind == "tokenize":
        needed = [tokenization_dir / f"pretrain_{suffix}.json" for suffix in ("2k", "4k", "8k")]
        needed += [tokenization_dir / "sft_general.json", tokenization_dir / "sft_tool.json"]
        return all(path.exists() for path in needed)
    if kind == "curriculum":
        return (Path(stage.get("out_dir", "checkpoints/curriculum_145m")) / "stage_02_seq8192" / "final" / "meta.json").exists()
    if kind in {"sft", "grpo", "opd"}:
        return (Path(stage.get("out_dir", f"checkpoints/{stage['id']}")) / "final" / "meta.json").exists()
    if kind == "eval":
        return Path(stage.get("out_path", f"reports/{stage['id']}.json")).exists()
    if kind == "export":
        return (Path(stage.get("out", "hf/minimind_145m")) / "config.json").exists()
    return False

def run_pipeline(config_path: str | Path, *, dry_run: bool = False, start: str | None = None, stop: str | None = None, force: bool = False) -> dict[str, Any]:
    spec = load_yaml(config_path)
    name = str(spec.get("name", "run"))
    run_dir = Path(spec.get("run_dir", f"runs/{name}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(spec.get("data_dir", "data"))
    tokenization_dir = Path(spec.get("tokenization_dir", data_dir / "tokenization"))
    nproc = int(os.environ.get("NPROC_PER_NODE", spec.get("nproc", 1)))
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
            (run_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
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
        elif kind == "export":
            _export_stage(stage, run_dir=run_dir, dry_run=dry_run)
        else:
            raise SystemExit(f"unknown stage kind {kind!r}")
        state["stages"][stage_id] = "dry-run" if dry_run else "done"
        (run_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
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
