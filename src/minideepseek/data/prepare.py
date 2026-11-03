"""Command-line data preparation entry point.

Examples::

    python -m minideepseek.data.prepare pretrain \
        --input ../minimind/dataset/pretrain_t2t_mini.jsonl \
        --output data/processed/pretrain/general/pilot.npz \
        --seq-len 4096 --max-records 2000

    python -m minideepseek.data.prepare fertility --input ...
    python -m minideepseek.data.prepare manifest --manifest data/manifests/pretrain_6b.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..tokenizer import load_tokenizer
from .contamination import build_eval_ngrams, find_contamination
from .fertility import fertility_report
from .grpo import build_grpo_dataset
from .io import iter_jsonl
from .manifest import load_manifest, validate_manifest
from .pretrain import build_pretrain_dataset
from .sft import build_sft_dataset


def _print(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _resolve(path: str) -> Path:
    return Path(path).expanduser()


def cmd_pretrain(args: argparse.Namespace) -> int:
    meta = build_pretrain_dataset(
        [_resolve(path) for path in args.input],
        _resolve(args.output),
        tokenizer=load_tokenizer(vocab_limit=args.vocab_limit),
        seq_len=args.seq_len,
        max_records=args.max_records,
        max_tokens=args.max_tokens,
        min_chars=args.min_chars,
        seed=args.seed,
        dedup=not args.no_dedup,
        quality_filter=not args.no_quality,
        domain=args.domain,
    )
    _print(meta)
    return 0


def cmd_sft(args: argparse.Namespace) -> int:
    meta = build_sft_dataset(
        [_resolve(path) for path in args.input],
        _resolve(args.output),
        tokenizer=load_tokenizer(vocab_limit=args.vocab_limit),
        seq_len=args.seq_len,
        max_records=args.max_records,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    _print(meta)
    return 0


def cmd_grpo(args: argparse.Namespace) -> int:
    meta = build_grpo_dataset(
        [_resolve(path) for path in args.input],
        _resolve(args.output),
        tokenizer=load_tokenizer(vocab_limit=args.vocab_limit),
        domain=args.domain,
        verifier=args.verifier,
        require_reference=not args.allow_missing_reference,
        max_records=args.max_records,
        seed=args.seed,
    )
    _print(meta)
    return 0


def cmd_opd(args: argparse.Namespace) -> int:
    from .opd import build_opd_pool

    teacher_keys = {}
    for spec in args.teacher_key or []:
        if "=" not in spec:
            raise SystemExit(f"--teacher-key expects name=path, got {spec!r}")
        key, path = spec.split("=", 1)
        teacher_keys[Path(path).stem] = key
    meta = build_opd_pool(
        [_resolve(path) for path in args.input],
        _resolve(args.output),
        teacher_keys=teacher_keys,
        default_teacher=args.default_teacher,
        max_per_source=args.max_per_source,
    )
    _print(meta)
    return 0


def cmd_fertility(args: argparse.Namespace) -> int:
    texts = (record.get("text", "") for record in iter_jsonl(_resolve(args.input)))
    report = fertility_report(texts, max_documents=args.limit)
    _print(report.as_dict())
    return 0


def cmd_contamination(args: argparse.Namespace) -> int:
    from .contamination import iter_eval_texts

    train_texts = (record.get("text") or record.get("content") or "" for record in iter_jsonl(_resolve(args.train)))
    eval_texts = list(iter_eval_texts(iter_jsonl(_resolve(args.eval))))
    eval_ngrams = build_eval_ngrams(eval_texts, n=args.n)
    matches = find_contamination(train_texts, eval_ngrams, n=args.n)
    _print(
        {
            "eval_ngrams": len(eval_ngrams),
            "contaminated_documents": len(matches),
            "examples": [
                {"index": match.index, "overlap": match.overlap, "ngram": " ".join(match.example)}
                for match in matches[:10]
            ],
        }
    )
    return 1 if matches else 0


def cmd_manifest(args: argparse.Namespace) -> int:
    manifest = load_manifest(_resolve(args.manifest))
    report = validate_manifest(manifest)
    _print(
        {
            "name": report.name,
            "ok": report.ok,
            "total_tokens": report.total_tokens,
            "seq_len": report.seq_len,
            "entries": [entry.__dict__ for entry in report.entries],
            "errors": report.errors,
            "warnings": report.warnings,
        }
    )
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain", help="tokenize + pack pretraining JSONL")
    pretrain.add_argument("--input", nargs="+", required=True)
    pretrain.add_argument("--output", required=True)
    pretrain.add_argument("--seq-len", type=int, default=4096)
    pretrain.add_argument("--max-records", type=int)
    pretrain.add_argument("--max-tokens", type=int)
    pretrain.add_argument("--min-chars", type=int, default=32)
    pretrain.add_argument("--domain", default="general")
    pretrain.add_argument("--seed", type=int, default=1234)
    pretrain.add_argument("--vocab-limit", type=int, default=None, help="debug-only token-id modulo")
    pretrain.add_argument("--no-dedup", action="store_true")
    pretrain.add_argument("--no-quality", action="store_true")
    pretrain.set_defaults(func=cmd_pretrain)

    sft = subparsers.add_parser("sft", help="tokenize SFT conversations with response masks")
    sft.add_argument("--input", nargs="+", required=True)
    sft.add_argument("--output", required=True)
    sft.add_argument("--seq-len", type=int, default=4096)
    sft.add_argument("--max-records", type=int)
    sft.add_argument("--max-tokens", type=int)
    sft.add_argument("--seed", type=int, default=1234)
    sft.add_argument("--vocab-limit", type=int, default=None, help="debug-only token-id modulo")
    sft.set_defaults(func=cmd_sft)

    grpo = subparsers.add_parser("grpo", help="convert prompts to GRPO JSONL")
    grpo.add_argument("--input", nargs="+", required=True)
    grpo.add_argument("--output", required=True)
    grpo.add_argument("--domain", default="math")
    grpo.add_argument("--verifier", default="math_exact_match")
    grpo.add_argument("--max-records", type=int)
    grpo.add_argument("--allow-missing-reference", action="store_true")
    grpo.add_argument("--seed", type=int, default=1234)
    grpo.add_argument("--vocab-limit", type=int, default=None, help="debug-only token-id modulo")
    grpo.set_defaults(func=cmd_grpo)

    opd = subparsers.add_parser("opd", help="build a mixed OPD prompt pool")
    opd.add_argument("--input", nargs="+", required=True)
    opd.add_argument("--output", required=True)
    opd.add_argument("--teacher-key", action="append", metavar="NAME=PATH")
    opd.add_argument("--default-teacher", default="general")
    opd.add_argument("--max-per-source", type=int)
    opd.set_defaults(func=cmd_opd)

    fertility = subparsers.add_parser("fertility", help="tokenizer fertility report")
    fertility.add_argument("--input", required=True)
    fertility.add_argument("--limit", type=int, default=1000)
    fertility.set_defaults(func=cmd_fertility)

    contamination = subparsers.add_parser("contamination", help="13-gram eval contamination check")
    contamination.add_argument("--train", required=True)
    contamination.add_argument("--eval", required=True)
    contamination.add_argument("--n", type=int, default=13)
    contamination.set_defaults(func=cmd_contamination)

    manifest = subparsers.add_parser("manifest", help="validate a dataset manifest")
    manifest.add_argument("--manifest", required=True)
    manifest.set_defaults(func=cmd_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
