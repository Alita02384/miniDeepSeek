#!/usr/bin/env python3
"""Export a miniDeepSeek training checkpoint to a Hugging Face directory.

    .venv/bin/python scripts/export_hf.py \
        --checkpoint checkpoints/pretrain_full/final \
        --out hf/mini_60m --flavor mini_60m --seq-len 8192

The result can be loaded with ``trust_remote_code=True`` by transformers,
verl, or any vLLM build that supports the model's auto_map.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minideepseek.model.hf.export import export_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--flavor", default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--tie-word-embeddings", action="store_true", default=None)
    parser.add_argument("--swiglu-limit", type=float, default=None)
    parser.add_argument("--index-topk", type=int, default=None)
    args = parser.parse_args()
    out = export_checkpoint(
        args.checkpoint,
        args.out,
        flavor=args.flavor,
        seq_len=args.seq_len,
        tie_word_embeddings=args.tie_word_embeddings,
        swiglu_limit=args.swiglu_limit,
        index_topk=args.index_topk,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
