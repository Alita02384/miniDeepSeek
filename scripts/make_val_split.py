#!/usr/bin/env python3
"""Carve an independent validation split out of a packed pretraining dataset.

The JSONL corpora behind ``data/tokenization`` have been deleted from the
training host (disk budget), so a new validation slice cannot be re-tokenized.
It does not need to be: every packed dataset keeps the *token stream* verbatim
in ``<stem>.input_ids.npy`` next to per-token ``positions``, whose ``== 0``
marks a document start.  A held-out slice is therefore just a token window of
that stream, re-packed at the sequence length of the stage that reads it.

Why this exists: the original validation sets are 1M tokens cut from the *head*
of each corpus -- 0.07% of the general corpus.  Measured on the 145M run they
are not representative: on ``val_general_4k`` one layer's router puts 0.8% of
its tokens on a single expert, while the training stream puts 17.2% there.  A
validation loss computed from 40 packs of that text moves more with the packing
than with model quality, and it cannot separate ``general`` from the other
domains at all.

Usage::

    # 40M-token general validation slice packed at 4K, taking those tokens out
    # of the 8K stage's training pool (run *before* that stage starts):
    python scripts/make_val_split.py \\
        --source data/tokenization/pretrain_general_8k.npz \\
        --tokens 40000000 --seq-len 4096 --domain general \\
        --out data/tokenization/val_general_4k_v2.npz --trim-source

Only the tail is ever carved, so ``--trim-source`` is a truncation: it never
moves tokens that a running stage has already consumed, which is what makes it
safe to do mid-run as long as the *source* dataset is not the current stage's.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minideepseek.data.io import load_flat_dataset, save_flat_dataset  # noqa: E402


def document_bounds(positions: np.ndarray) -> list[tuple[int, int]]:
    """Split a flat ``positions`` array into ``(start, stop)`` document spans."""
    starts = np.flatnonzero(np.asarray(positions) == 0).tolist()
    if not starts or starts[0] != 0:
        # A window that begins mid-document: the cut itself opens a "document",
        # exactly like the packer's own truncation of an over-long document.
        starts = [0, *starts]
    stops = [*starts[1:], int(len(positions))]
    return list(zip(starts, stops, strict=True))


def pack_documents(
    stream: np.ndarray, bounds: list[tuple[int, int]], seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedily pack documents, byte-compatible with ``data.pretrain.pack_documents``.

    Same rules: at most ``seq_len`` tokens per pack, a document longer than
    ``seq_len`` truncated rather than split, position ids restart at 0 for every
    document, and ``target_mask`` is false only at a document's first token.
    """
    packs_ids: list[np.ndarray] = []
    packs_pos: list[np.ndarray] = []
    offsets: list[int] = [0]
    pending_ids: list[np.ndarray] = []
    pending_pos: list[np.ndarray] = []
    used = 0

    def flush() -> None:
        nonlocal pending_ids, pending_pos, used
        if not pending_ids:
            return
        ids = np.concatenate(pending_ids)
        packs_ids.append(ids)
        packs_pos.append(np.concatenate(pending_pos))
        offsets.append(offsets[-1] + int(ids.size))
        pending_ids, pending_pos, used = [], [], 0

    for start, stop in bounds:
        doc = stream[start:stop]
        if doc.size > seq_len:
            doc = doc[:seq_len]
        if doc.size == 0:
            continue
        if used and used + doc.size > seq_len:
            flush()
        pending_ids.append(doc)
        pending_pos.append(np.arange(doc.size, dtype=np.int32))
        used += doc.size
    flush()

    if not packs_ids:
        raise SystemExit("the requested window holds no tokens")
    positions = np.concatenate(packs_pos)
    return (
        np.concatenate(packs_ids),
        positions,
        positions != 0,
        np.asarray(offsets, dtype=np.int64),
    )


def _tokenizer_fields(meta: dict) -> dict:
    return {
        key: meta[key]
        for key in ("tokenizer_hash", "tokenizer_vocab", "format")
        if key in meta
    }


def write_val_split(
    source: Path,
    *,
    tokens: int,
    seq_len: int,
    domain: str,
    out: Path,
) -> dict:
    meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    input_ids, positions, _, _ = load_flat_dataset(source)
    total = int(len(input_ids))
    if tokens <= 0 or tokens >= total:
        raise SystemExit(f"--tokens must be in (0, {total}), got {tokens}")
    lo = total - tokens
    window_ids = np.asarray(input_ids[lo:], dtype=np.int32)
    window_positions = np.asarray(positions[lo:], dtype=np.int32)
    bounds = document_bounds(window_positions)
    packed = pack_documents(window_ids, bounds, seq_len)
    documents = len(bounds)
    packed_ids, packed_positions, packed_mask, offsets = packed
    # ``pack_documents`` truncates a document longer than ``seq_len`` instead of
    # splitting it, so a window of over-long documents yields fewer packed
    # tokens than asked for.  Silent loss is how a "40M-token" validation set
    # ends up being 10M, so say it out loud.
    yield_ratio = packed_ids.size / tokens
    if yield_ratio < 0.999:
        print(
            f"  note: {tokens - int(packed_ids.size)} tokens were clipped by "
            f"documents longer than {seq_len} ({yield_ratio:.1%} of the window kept)"
        )
    val_meta = {
        **_tokenizer_fields(meta),
        "seq_len": int(seq_len),
        "packs": int(len(offsets) - 1),
        "tokens": int(packed_ids.size),
        "target_tokens": int(packed_mask.sum()),
        "documents_kept": int(documents),
        "domain": domain,
        "sources": list(meta.get("sources", [])),
        "skip_tokens": int(meta.get("skip_tokens", 0)) + lo,
        "max_tokens": int(tokens),
        "streaming": True,
        # Provenance: which dataset this window was carved out of, and where.
        "split": "tail_holdout",
        "split_of": source.name,
        "split_offset_tokens": lo,
        "split_yield_ratio": round(yield_ratio, 6),
    }
    save_flat_dataset(
        out,
        input_ids=packed_ids,
        positions=packed_positions,
        target_mask=packed_mask,
        offsets=offsets,
        meta=val_meta,
    )
    return val_meta


def trim_source(source: Path, tokens: int) -> dict:
    """Rewrite ``source`` without its last ``tokens`` tokens.

    Two things make this delicate enough to be worth spelling out:

    * the arrays being written are memory-mapped *from* the files being
      replaced, so writing in place is not an option -- ``np.save``'s
      ``open(..., "wb")`` truncates the input before a single token is copied
      out (this destroyed a 738M-token general slice once);
    * the staged files must keep the *source file name*, because
      ``save_flat_dataset`` derives its sidecar names with ``with_suffix`` --
      staging to ``<source>.tmp`` would rewrite the destination namespace
      instead of a temporary one.

    So: copy into a sibling staging directory under the real name, then
    ``os.replace`` each sidecar into place.
    """
    meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    input_ids, positions, target_mask, offsets = load_flat_dataset(source)
    total = int(len(input_ids))
    keep = total - tokens
    kept_offsets = offsets[offsets <= keep]
    if kept_offsets.size == 0 or int(kept_offsets[-1]) != keep:
        kept_offsets = np.append(kept_offsets, keep)
    trimmed = {
        **meta,
        "packs": int(len(kept_offsets) - 1),
        "tokens": int(keep),
        "target_tokens": int(np.array(target_mask[:keep], dtype=bool).sum()),
        "max_tokens": int(keep),
        "trimmed_tail_tokens": int(tokens),
        "trimmed_tail_for": f"{source.stem}_val",
    }
    staging = source.parent / f".{source.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    staged = staging / source.name
    # ``copy=True`` matters: ``np.asarray`` on a memmap slice returns a view, so
    # the buffer would still be the file the staging write is about to touch.
    save_flat_dataset(
        staged,
        input_ids=np.array(input_ids[:keep], dtype=np.int32, copy=True),
        positions=np.array(positions[:keep], dtype=np.int32, copy=True),
        target_mask=np.array(target_mask[:keep], dtype=bool, copy=True),
        offsets=np.array(kept_offsets, dtype=np.int64, copy=True),
        meta=trimmed,
    )
    # Reuse the loader's own naming rule (``X.npz`` -> ``X.<name>.npy``) instead
    # of rebuilding it: ``Path.with_suffix`` replaces the ``.npz`` extension, so
    # a hand-written ``<source.name>.<name>.npy`` silently lands one level off.
    for name in ("input_ids", "positions", "target_mask", "offsets"):
        final = source.with_suffix(f".{name}.npy")
        (staging / final.name).replace(final)
    final_meta = source.with_suffix(".json")
    (staging / final_meta.name).replace(final_meta)
    shutil.rmtree(staging, ignore_errors=True)
    return trimmed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="packed dataset to carve")
    parser.add_argument("--tokens", type=int, required=True, help="holdout size")
    parser.add_argument("--seq-len", type=int, help="target packing (omit for --trim-only)")
    parser.add_argument("--domain", default="general")
    parser.add_argument("--out", help="validation dataset path")
    parser.add_argument(
        "--trim-source",
        action="store_true",
        help="also rewrite the source without the holdout (only for a stage "
        "that has not started yet: the token stream must not have been read)",
    )
    parser.add_argument(
        "--trim-only",
        action="store_true",
        help="skip the validation write and only drop the tail from the source. "
        "Use after the per-length validation sets have been written, so the "
        "second invocation still sees the untrimmed tail.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.trim_only and (args.seq_len is None or args.out is None):
        parser.error("--seq-len and --out are required unless --trim-only")

    source = Path(args.source)
    meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    total = int(meta.get("tokens", 0))
    print(
        f"{source.name}: {total:,} tokens -> holdout {args.tokens:,} "
        f"({args.tokens / max(total, 1):.2%}), train keeps {total - args.tokens:,}"
    )
    if args.dry_run:
        print("dry run: no files written")
        return 0
    if not args.trim_only:
        out = Path(args.out)
        val_meta = write_val_split(
            source,
            tokens=args.tokens,
            seq_len=int(args.seq_len),
            domain=args.domain,
            out=out,
        )
        print(
            f"wrote {out}: {val_meta['tokens']:,} tokens in "
            f"{val_meta['packs']:,} packs of <= {args.seq_len}, "
            f"{val_meta['documents_kept']:,} documents"
        )
    if args.trim_source or args.trim_only:
        trimmed = trim_source(source, args.tokens)
        print(
            f"trimmed {source.name}: {trimmed['tokens']:,} tokens in "
            f"{trimmed['packs']:,} packs (was {total:,})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
