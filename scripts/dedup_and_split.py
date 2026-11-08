"""Exact-dedup, merge and down-sample a JSONL corpus to a token budget.

Two passes over the inputs:

1. **count** — hash each document's normalized text (blake2b-128), drop
   duplicates *across all inputs*, and record characters per source;
2. **write** — repeat the same walk and emit the surviving documents, keeping
   a deterministic hash-selected fraction when the corpus exceeds the budget.

Documents from several inputs are interleaved round-robin so a merged file does
not put one source entirely before another.  Tokens are estimated with measured
``tok/char`` ratios (``--ratio`` per source) because the tokenizer is not run
here — packing happens on the training host.

Examples::

    # general = MiniMind + external, target 2.46B tokens
    python scripts/dedup_and_split.py \
        --input mini:data/pretrain_t2t.jsonl ext:data/pretrain_general.jsonl \
        --ratio mini=0.743 ext=0.335 --target-tokens 2460000000 \
        --out data/pretrain_general.jsonl

    # code: 923M -> 450M
    python scripts/dedup_and_split.py --input code:data/pretrain_code.jsonl \
        --ratio code=0.432 --target-tokens 450000000 --out data/pretrain_code.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MASK = (1 << 62) - 1


def normalized(text: str) -> str:
    return " ".join(text.split())


def digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def streams(specs: list[tuple[str, Path]]):
    handles = [(name, path.open(encoding="utf-8")) for name, path in specs]
    live = [name for name, _ in handles]
    try:
        while live:
            for name, handle in handles:
                if name not in live:
                    continue
                line = handle.readline()
                if not line:
                    live.remove(name)
                    continue
                yield name, line
    finally:
        for _, handle in handles:
            handle.close()


def document_text(record: dict[str, Any]) -> str:
    value = record.get("text") or record.get("content") or ""
    return value if isinstance(value, str) else ""


def count_pass(specs, ratios) -> dict[str, Any]:
    seen: set[int] = set()
    chars: dict[str, int] = {name: 0 for name, _ in specs}
    docs: dict[str, int] = {name: 0 for name, _ in specs}
    dup = 0
    total = 0
    for name, line in streams(specs):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = document_text(record)
        if not text:
            continue
        total += 1
        key = digest(normalized(text))
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        docs[name] += 1
        chars[name] += len(text)
    tokens = sum(chars[name] * ratios[name] for name in chars)
    return {
        "documents": total,
        "duplicates": dup,
        "kept_documents": docs,
        "kept_chars": chars,
        "estimated_tokens": tokens,
    }


def write_pass(specs, ratios, out: Path, target_tokens: int | None, stats: dict) -> dict:
    ratio = 1.0
    if target_tokens and stats["estimated_tokens"] > target_tokens:
        ratio = target_tokens / stats["estimated_tokens"]
    threshold = int(ratio * MASK)
    seen: set[int] = set()
    kept = dropped_dup = dropped_sample = 0
    chars = 0
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as handle:
        for name, line in streams(specs):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = document_text(record)
            if not text:
                continue
            key = digest(normalized(text))
            if key in seen:
                dropped_dup += 1
                continue
            seen.add(key)
            if (key & MASK) > threshold:
                dropped_sample += 1
                continue
            record.setdefault("source", name)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            chars += len(text)
    tmp.replace(out)
    tokens = sum(
        stats["kept_chars"][name] * ratios[name] * ratio for name in stats["kept_chars"]
    )
    return {
        "output": str(out),
        "kept": kept,
        "dropped_duplicate": dropped_dup,
        "dropped_by_budget": dropped_sample,
        "keep_ratio": ratio,
        "chars": chars,
        "estimated_tokens": tokens,
    }


def parse_spec(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"expected name:path, got {value!r}")
    name, path = value.split(":", 1)
    return name, Path(path)


def parse_ratio(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected name=ratio, got {value!r}")
    name, ratio = value.split("=", 1)
    return name, float(ratio)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=parse_spec, required=True)
    parser.add_argument("--ratio", nargs="+", type=parse_ratio, required=True)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    ratios = dict(args.ratio)
    specs = list(args.input)
    missing = [name for name, _ in specs if name not in ratios]
    if missing:
        raise SystemExit(f"missing --ratio for {missing}")

    stats = count_pass(specs, ratios)
    print("count:", json.dumps(stats, ensure_ascii=False), flush=True)
    result = write_pass(specs, ratios, Path(args.out), args.target_tokens, stats)
    print("write:", json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
