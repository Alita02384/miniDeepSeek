"""Tokenize the flat JSONL dataset into mmap-able training artifacts.

Run on the training host, not in the local dev checkout: the full 2.3B-token
pretrain corpus produces roughly 20 GB per sequence length.  The JSONL files in
``data/`` remain the canonical, uploadable dataset; ``data/tokenization/`` is
generated and excluded from the release package.

Examples::

    # full corpus, 2K/4K/8K + general/tool SFT
    python -m minideepseek.data.tokenize --data-dir data --out-dir data/tokenization

    # CPU smoke
    python -m minideepseek.data.tokenize --smoke --max-records 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..tokenizer import load_tokenizer
from .pretrain import build_pretrain_dataset
from .sft import build_sft_dataset

SEQ_SUFFIX = {2048: "2k", 4096: "4k", 8192: "8k"}


def tokenize_pretrain(
    input_path: Path,
    out_dir: Path,
    *,
    seq_lens: list[int],
    max_records: int | None = None,
    max_tokens: int | None = None,
    dedup: bool = False,
    paragraph_dedup: bool = False,
    quality_filter: bool = True,
) -> dict:
    tokenizer = load_tokenizer()
    reports = {}
    for seq_len in seq_lens:
        output = out_dir / f"pretrain_{SEQ_SUFFIX.get(seq_len, str(seq_len))}.npz"
        reports[str(seq_len)] = build_pretrain_dataset(
            [input_path],
            output,
            tokenizer=tokenizer,
            seq_len=seq_len,
            max_records=max_records,
            max_tokens=max_tokens,
            dedup=dedup,
            paragraph_dedup=paragraph_dedup,
            quality_filter=quality_filter,
            domain="general",
        )
    return reports


def tokenize_sft(
    input_path: Path,
    output_path: Path,
    *,
    seq_len: int,
    max_records: int | None = None,
    max_tokens: int | None = None,
) -> dict:
    return build_sft_dataset(
        [input_path],
        output_path,
        tokenizer=load_tokenizer(),
        seq_len=seq_len,
        max_records=max_records,
        max_tokens=max_tokens,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="data/tokenization")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--sft-seq-len", type=int, default=4096)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dedup", action="store_true", help="enable built-in MinHash (debug only)")
    parser.add_argument("--paragraph-dedup", action="store_true")
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU-sized run")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_records = args.max_records
    if args.smoke and max_records is None:
        max_records = 200

    report: dict = {}
    if not args.skip_pretrain:
        report["pretrain"] = tokenize_pretrain(
            data_dir / "pretrain_t2t.jsonl",
            out_dir,
            seq_lens=args.seq_lens,
            max_records=max_records,
            max_tokens=args.max_tokens,
            dedup=args.dedup,
            paragraph_dedup=args.paragraph_dedup,
            quality_filter=not args.no_quality,
        )
    if not args.skip_sft:
        report["sft_general"] = tokenize_sft(
            data_dir / "sft_general.jsonl",
            out_dir / "sft_general.npz",
            seq_len=args.sft_seq_len,
            max_records=max_records,
            max_tokens=args.max_tokens,
        )
        report["sft_tool"] = tokenize_sft(
            data_dir / "sft_tool.jsonl",
            out_dir / "sft_tool.npz",
            seq_len=args.sft_seq_len,
            max_records=max_records,
            max_tokens=args.max_tokens,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
