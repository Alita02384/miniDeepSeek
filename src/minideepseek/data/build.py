"""Build the flat, uploadable dataset from the MiniMind data checkout.

The release layout is intentionally flat::

    data/pretrain_t2t.jsonl
    data/sft_general.jsonl
    data/sft_tool.jsonl
    data/grpo_general_train.jsonl  data/eval_general.jsonl
    data/grpo_tool_train.jsonl     data/eval_tool.jsonl
    data/opd_prompts_general.jsonl data/opd_prompts_tool.jsonl
    data/opd_prompts_mixed.jsonl
    data/tokenization/             # generated later on the training host
    data/manifests/                # small provenance/stat files

Only files that need transformation are rewritten; pretrain and general SFT
are copied/hardlinked verbatim.  ``minimind/dataset`` is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from ..tokenizer import load_tokenizer, normalize_conversation
from .io import iter_jsonl, write_jsonl
from .sft import encode_sft_record

DEFAULT_SEQ_LEN = 4096


def _preview(path: Path) -> str:
    return str(path)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def copy_file(source: Path, target: Path, *, force: bool = False, dry_run: bool = False) -> str:
    _ensure_parent(target)
    if target.exists() and not force:
        return "kept"
    if dry_run:
        return f"would copy {source} -> {target}"
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
        return "hardlinked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def _messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    conversations = record.get("conversations") or record.get("messages") or []
    return normalize_conversation({"messages": conversations}) if conversations else []


def _has_tool(record: dict[str, Any]) -> bool:
    for message in record.get("conversations") or []:
        if not isinstance(message, dict):
            continue
        if message.get("tools") or message.get("tool_calls"):
            return True
    return False


def _tools_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for message in messages:
        tools = message.get("tools")
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                tools = None
        if isinstance(tools, list):
            return tools
    return []


def _strip_trailing_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = [dict(message) for message in messages]
    while messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return messages


def _prompt_key(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split(records: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import random

    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    n_val = int(len(records) * val_ratio)
    val_indices = set(order[:n_val])
    train = [record for index, record in enumerate(records) if index not in val_indices]
    val = [record for index, record in enumerate(records) if index in val_indices]
    return train, val


def build_sft_general(source: Path, target: Path, *, force: bool, dry_run: bool) -> dict[str, Any]:
    action = copy_file(source, target, force=force, dry_run=dry_run)
    return {"source": _preview(source), "target": _preview(target), "action": action}


def build_sft_tool(
    source: Path,
    target: Path,
    *,
    tokenizer,
    seq_len: int,
    general_ratio: float,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Select tool records from MiniMind SFT and mix in a small general slice."""
    if target.exists() and not force:
        return {"target": _preview(target), "action": "kept"}
    if dry_run:
        return {"target": _preview(target), "action": "would build"}

    tool_records: list[dict[str, Any]] = []
    tool_tokens = 0
    for record in iter_jsonl(source):
        if not _has_tool(record):
            continue
        try:
            encoded = encode_sft_record(record, tokenizer, seq_len=seq_len)
        except ValueError:
            continue
        if encoded is None:
            continue
        tool_records.append(record)
        tool_tokens += len(encoded[0])

    general_target = int(tool_tokens * max(general_ratio, 0.0))
    selected_general: list[dict[str, Any]] = []
    general_tokens = 0
    if general_target > 0:
        for record in iter_jsonl(source):
            if _has_tool(record):
                continue
            if general_tokens >= general_target:
                break
            try:
                encoded = encode_sft_record(record, tokenizer, seq_len=seq_len)
            except ValueError:
                continue
            if encoded is None:
                continue
            selected_general.append(record)
            general_tokens += len(encoded[0])

    write_jsonl(target, tool_records + selected_general)
    return {
        "target": _preview(target),
        "tool_records": len(tool_records),
        "general_records": len(selected_general),
        "tool_tokens": tool_tokens,
        "general_tokens": general_tokens,
    }


def build_grpo_general(
    source: Path,
    train_target: Path,
    eval_target: Path,
    *,
    val_ratio: float,
    seed: int,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """MiniMind RLAIF prompts: drop the trailing empty assistant turn."""
    if train_target.exists() and eval_target.exists() and not force:
        return {"train": _preview(train_target), "eval": _preview(eval_target), "action": "kept"}
    if dry_run:
        return {"train": _preview(train_target), "action": "would build"}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in iter_jsonl(source):
        messages = _strip_trailing_assistant(_messages(record))
        if not messages:
            continue
        key = _prompt_key(messages)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "prompt": messages,
                "domain": "general",
                "verifier": "none",
                "multi_turn": True,
                "source": "rlaif",
                "license": "unknown",
            }
        )
    train, val = _split(records, val_ratio, seed)
    write_jsonl(train_target, train)
    write_jsonl(eval_target, val)
    return {"train": _preview(train_target), "eval": _preview(eval_target), "records": len(records), "train_records": len(train), "eval_records": len(val)}


def build_grpo_tool(
    sources: Iterable[Path],
    train_target: Path,
    eval_target: Path,
    *,
    val_ratio: float,
    seed: int,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """MiniMind ``agent_rl*.jsonl`` -> multi-turn tool GRPO records."""
    if train_target.exists() and eval_target.exists() and not force:
        return {"train": _preview(train_target), "eval": _preview(eval_target), "action": "kept"}
    if dry_run:
        return {"train": _preview(train_target), "action": "would build"}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        for record in iter_jsonl(source):
            messages = _strip_trailing_assistant(_messages(record))
            tools = _tools_from_messages(messages)
            gt = [str(value) for value in (record.get("gt") or []) if str(value).strip()]
            if not messages or not tools or not gt:
                continue
            key = _prompt_key(messages)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "prompt": messages,
                    "domain": "tool",
                    "verifier": "tool",
                    "multi_turn": True,
                    "tools": tools,
                    "gt": gt,
                    "reference_answer": gt[0],
                    "source": source.stem,
                    "license": "unknown",
                }
            )
    train, val = _split(records, val_ratio, seed)
    write_jsonl(train_target, train)
    write_jsonl(eval_target, val)
    return {"train": _preview(train_target), "eval": _preview(eval_target), "records": len(records), "train_records": len(train), "eval_records": len(val)}




def build_sft_code(source: Path, target: Path, *, force: bool, dry_run: bool) -> dict[str, Any]:
    """Copy a code instruction JSONL (conversations/messages) into the flat set.

    Code SFT data is external; the accepted format is the same conversation
    schema as ``sft_general``.  We do not rewrite records here because the
    upstream code corpus is still being selected.
    """
    if not source.exists():
        return {"target": _preview(target), "missing": True, "source": _preview(source)}
    action = copy_file(source, target, force=force, dry_run=dry_run)
    return {"source": _preview(source), "target": _preview(target), "action": action}


def build_grpo_code(
    source: Path,
    train_target: Path,
    eval_target: Path,
    *,
    val_ratio: float,
    seed: int,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Convert code RL records with ``rubric.tests`` into GRPO JSONL."""
    if not source.exists():
        return {"train": _preview(train_target), "eval": _preview(eval_target), "missing": True}
    if train_target.exists() and eval_target.exists() and not force:
        return {"train": _preview(train_target), "eval": _preview(eval_target), "action": "kept"}
    if dry_run:
        return {"train": _preview(train_target), "action": "would build"}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in iter_jsonl(source):
        prompt = record.get("prompt") or record.get("messages")
        if not prompt and record.get("conversations"):
            prompt = _strip_trailing_assistant(_messages(record))
        if not isinstance(prompt, list) or not prompt:
            continue
        rubric = dict(record.get("rubric") or {})
        tests = rubric.get("tests") or rubric.get("test_code") or record.get("tests") or record.get("test_code")
        if not tests:
            continue
        key = _prompt_key(prompt)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "prompt": prompt,
                "domain": "code",
                "verifier": "unit_test",
                "rubric": {**rubric, "tests": tests},
                "multi_turn": False,
                "source": record.get("source", source.stem),
                "license": record.get("license", "unknown"),
            }
        )
    train, val = _split(records, val_ratio, seed)
    write_jsonl(train_target, train)
    write_jsonl(eval_target, val)
    return {"train": _preview(train_target), "eval": _preview(eval_target), "records": len(records), "train_records": len(train), "eval_records": len(val)}

def build_opd_prompts(
    general_source: Path,
    tool_source: Path,
    output_dir: Path,
    *,
    code_source: Path | None = None,
    max_per_domain: int,
    seed: int,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return {"action": "would build"}
    import random

    per_domain: dict[str, list[dict[str, Any]]] = {}
    sources = [("general", general_source), ("tool", tool_source)]
    if code_source is not None and code_source.exists():
        sources.append(("code", code_source))
    for domain, source in sources:
        prompts = []
        for record in iter_jsonl(source):
            prompt = record.get("prompt")
            if not prompt:
                continue
            prompts.append(
                {
                    "prompt": prompt,
                    "teacher_key": domain,
                    "domain": domain,
                    "source": record.get("source", domain),
                    "multi_turn": bool(record.get("multi_turn", False)),
                }
            )
        random.Random(seed).shuffle(prompts)
        per_domain[domain] = prompts[:max_per_domain]
        write_jsonl(output_dir / f"opd_prompts_{domain}.jsonl", per_domain[domain])

    # Interleave domains so every OPD step sees a balanced batch.
    mixed: list[dict[str, Any]] = []
    keys = sorted(per_domain)
    for index in range(max((len(values) for values in per_domain.values()), default=0)):
        for key in keys:
            if index < len(per_domain[key]):
                mixed.append(per_domain[key][index])
    write_jsonl(output_dir / "opd_prompts_mixed.jsonl", mixed)
    return {"mixed": len(mixed), "per_domain": {key: len(value) for key, value in per_domain.items()}}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_all(
    minimind_root: Path,
    output_dir: Path,
    *,
    seed: int = 1234,
    val_ratio: float = 0.1,
    general_ratio: float = 0.25,
    max_per_domain: int = 20000,
    seq_len: int = DEFAULT_SEQ_LEN,
    force: bool = False,
    dry_run: bool = False,
    skip_pretrain: bool = False,
    code_sft: Path | None = None,
    code_grpo: Path | None = None,
    code_eval: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenization_dir = output_dir / "tokenization"
    manifest_dir = output_dir / "manifests"
    tokenization_dir.mkdir(exist_ok=True)
    manifest_dir.mkdir(exist_ok=True)
    tokenizer = load_tokenizer()
    report: dict[str, Any] = {}

    if not skip_pretrain:
        report["pretrain"] = copy_file(
            minimind_root / "pretrain_t2t.jsonl",
            output_dir / "pretrain_t2t.jsonl",
            force=force,
            dry_run=dry_run,
        )
    report["sft_general"] = build_sft_general(
        minimind_root / "sft_t2t_mini.jsonl",
        output_dir / "sft_general.jsonl",
        force=force,
        dry_run=dry_run,
    )
    report["sft_tool"] = build_sft_tool(
        output_dir / "sft_general.jsonl",
        output_dir / "sft_tool.jsonl",
        tokenizer=tokenizer,
        seq_len=seq_len,
        general_ratio=general_ratio,
        force=force,
        dry_run=dry_run,
    )
    report["grpo_general"] = build_grpo_general(
        minimind_root / "rlaif.jsonl",
        output_dir / "grpo_general_train.jsonl",
        output_dir / "eval_general.jsonl",
        val_ratio=val_ratio,
        seed=seed,
        force=force,
        dry_run=dry_run,
    )
    report["grpo_tool"] = build_grpo_tool(
        [minimind_root / "agent_rl.jsonl", minimind_root / "agent_rl_math.jsonl"],
        output_dir / "grpo_tool_train.jsonl",
        output_dir / "eval_tool.jsonl",
        val_ratio=val_ratio,
        seed=seed,
        force=force,
        dry_run=dry_run,
    )
    missing: list[str] = []
    if code_sft is not None:
        report["sft_code"] = build_sft_code(
            code_sft, output_dir / "sft_code.jsonl", force=force, dry_run=dry_run
        )
        if report["sft_code"].get("missing"):
            missing.append("sft_code")
    else:
        report["sft_code"] = {"missing": True, "reason": "--code-sft not provided"}
        missing.append("sft_code")
    if code_grpo is not None:
        report["grpo_code"] = build_grpo_code(
            code_grpo,
            output_dir / "grpo_code_train.jsonl",
            output_dir / "eval_code.jsonl",
            val_ratio=val_ratio,
            seed=seed,
            force=force,
            dry_run=dry_run,
        )
        if report["grpo_code"].get("missing"):
            missing.append("grpo_code")
    else:
        report["grpo_code"] = {"missing": True, "reason": "--code-grpo not provided"}
        missing.append("grpo_code")
    if code_eval is not None:
        report["eval_code"] = build_sft_code(
            code_eval, output_dir / "eval_code.jsonl", force=force, dry_run=dry_run
        )
        if report["eval_code"].get("missing"):
            missing.append("eval_code")
    elif not (output_dir / "eval_code.jsonl").exists():
        missing.append("eval_code")

    report["opd"] = build_opd_prompts(
        output_dir / "grpo_general_train.jsonl",
        output_dir / "grpo_tool_train.jsonl",
        output_dir,
        code_source=output_dir / "grpo_code_train.jsonl",
        max_per_domain=max_per_domain,
        seed=seed,
        force=force,
        dry_run=dry_run,
    )

    if not dry_run:
        stats: dict[str, Any] = {}
        for path in sorted(output_dir.glob("*.jsonl")):
            stats[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        (manifest_dir / "stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_manifest(
            manifest_dir / "specialists.yaml",
            {
                "name": "minimind_specialists",
                "seed": seed,
                "val_ratio": val_ratio,
                "general_ratio": general_ratio,
                "sources": {
                    "pretrain": str(minimind_root / "pretrain_t2t.jsonl"),
                    "sft_general": str(minimind_root / "sft_t2t_mini.jsonl"),
                    "sft_tool": str(minimind_root / "sft_t2t_mini.jsonl"),
                    "grpo_general": str(minimind_root / "rlaif.jsonl"),
                    "grpo_tool": [
                        str(minimind_root / "agent_rl.jsonl"),
                        str(minimind_root / "agent_rl_math.jsonl"),
                    ],
                    "code_sft": str(code_sft) if code_sft else None,
                    "code_grpo": str(code_grpo) if code_grpo else None,
                    "code_eval": str(code_eval) if code_eval else None,
                },
                "missing": sorted(set(missing)),
                "notes": [
                    "sft_general is copied verbatim from sft_t2t_mini.jsonl",
                    "sft_tool is the tool/tool_call subset plus a general slice",
                    "grpo_general is the rlaif prompt pool with the trailing empty assistant dropped",
                    "grpo_tool is the agent_rl/agent_rl_math tool+gt subset, multi-turn",
                    "code expert is intentionally absent until code SFT/GRPO data is added",
                ],
            },
        )
    report["stats"] = _preview(manifest_dir / "stats.json")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimind-root", default="../minimind/dataset")
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--general-ratio", type=float, default=0.25)
    parser.add_argument("--max-per-domain", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--code-sft", default=None, help="code SFT JSONL (conversations)")
    parser.add_argument("--code-grpo", default=None, help="code GRPO JSONL (prompt + rubric.tests)")
    parser.add_argument("--code-eval", default=None, help="code eval JSONL")
    args = parser.parse_args(argv)
    report = build_all(
        Path(args.minimind_root).expanduser(),
        Path(args.out),
        seed=args.seed,
        val_ratio=args.val_ratio,
        general_ratio=args.general_ratio,
        max_per_domain=args.max_per_domain,
        seq_len=args.seq_len,
        force=args.force,
        dry_run=args.dry_run,
        skip_pretrain=args.skip_pretrain,
        code_sft=Path(args.code_sft).expanduser() if args.code_sft else None,
        code_grpo=Path(args.code_grpo).expanduser() if args.code_grpo else None,
        code_eval=Path(args.code_eval).expanduser() if args.code_eval else None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
