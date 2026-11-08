"""Stream a JSONL corpus into memory-mapped packed arrays for pretraining.

``minideepseek.data.pretrain.build_pretrain_dataset`` buffers the whole corpus
in RAM (~18 GB for 2B tokens).  This node has 16 GB, so the 3B-token build needs
a streaming writer: tokenize one document at a time, append to ``.npy``
sidecars, and patch the ``.npy`` header at the end.  The output is byte-for-byte
compatible with :func:`minideepseek.data.io.load_flat_dataset`.

Packing semantics match ``pack_documents``: greedy document packing, packs of
at most ``seq_len`` tokens, per-document position ids (so ``positions == 0``
marks a document start), and ``target_mask`` false only for each document's
first token.

Example::

    python scripts/pack_domain.py \
        --input data/pretrain_code.jsonl \
        --out data/tokenization/pretrain_code_4k.npz \
        --seq-len 4096 --skip-tokens 2000000 --max-tokens 450000000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from minideepseek.data.io import iter_jsonl
from minideepseek.tokenizer import load_tokenizer

MAGIC = b"\x93NUMPY"
HEADER_PAD = 128


class NpyStreamWriter:
    """Append-only ``.npy`` writer with an O(1) memory footprint."""

    def __init__(self, path: Path, dtype: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dtype = np.dtype(dtype)
        self.count = 0
        self._handle = self.path.open("wb")
        self._handle.write(MAGIC + b"\x01\x00" + HEADER_PAD.to_bytes(2, "little"))
        self._handle.write(b" " * HEADER_PAD)

    def write(self, values) -> None:
        array = np.ascontiguousarray(values, dtype=self.dtype)
        self._handle.write(array.tobytes())
        self.count += int(array.size)

    def close(self) -> None:
        header = (
            f"{{'descr': '{self.dtype.str}', 'fortran_order': False, "
            f"'shape': ({self.count},), }}"
        ).encode()
        if len(header) > HEADER_PAD - 1:
            raise RuntimeError(f"npy header too long for {self.path}: {header!r}")
        padded = header + b" " * (HEADER_PAD - len(header) - 1) + b"\n"
        self._handle.seek(10)
        self._handle.write(padded)
        self._handle.close()


def _encoder_text(record: dict) -> str:
    text = record.get("text") or record.get("content") or ""
    return text if isinstance(text, str) else ""


def pack(
    inputs: list[Path],
    out: Path,
    *,
    seq_len: int,
    skip_tokens: int,
    max_tokens: int | None,
    domain: str,
) -> dict:
    tokenizer = load_tokenizer()
    out = Path(out)
    stem = out.with_suffix("")
    ids_writer = NpyStreamWriter(stem.with_suffix(".input_ids.npy"), "int32")
    pos_writer = NpyStreamWriter(stem.with_suffix(".positions.npy"), "int32")
    mask_writer = NpyStreamWriter(stem.with_suffix(".target_mask.npy"), "bool")

    offsets: list[int] = [0]
    buf_ids: list[int] = []
    buf_pos: list[int] = []
    buf_mask: list[bool] = []
    total = 0
    packs = 0
    docs = 0
    skipped_tokens = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal buf_ids, buf_pos, buf_mask, total, packs
        if not buf_ids:
            return
        ids_writer.write(buf_ids)
        pos_writer.write(buf_pos)
        mask_writer.write(buf_mask)
        total += len(buf_ids)
        offsets.append(total)
        packs += 1
        buf_ids, buf_pos, buf_mask = [], [], []

    stop = False
    streams = [iter(iter_jsonl(path)) for path in inputs]

    def documents():
        # Round-robin over inputs so a multi-source domain (general = MiniMind +
        # external) interleaves instead of filling the budget from the first one.
        if len(streams) == 1:
            yield from streams[0]
            return
        live = list(range(len(streams)))
        while live:
            for index in list(live):
                try:
                    yield next(streams[index])
                except StopIteration:
                    live.remove(index)

    for record in documents():
        if stop:
            break
        if not isinstance(record, dict):
            continue
        text = _encoder_text(record)
        if not text:
            continue
        ids = tokenizer.encode(text)
        if not ids:
            continue
        if skipped_tokens < skip_tokens:
            skipped_tokens += len(ids)
            continue
        docs += 1
        if len(ids) > seq_len:
            ids = ids[:seq_len]
        if buf_ids and len(buf_ids) + len(ids) > seq_len:
            flush()
        for position, token in enumerate(ids):
            buf_ids.append(int(token))
            buf_pos.append(position)
            buf_mask.append(position != 0)
        if max_tokens is not None and total + len(buf_ids) >= max_tokens:
            flush()
            stop = True
            break
        if docs % 50_000 == 0:
            elapsed = time.perf_counter() - started
            rate = total / max(elapsed, 1e-9) / 1e6
            print(
                f"  {docs} docs, {total/1e6:.1f}M tokens, {rate:.2f}M tok/s",
                flush=True,
            )
    flush()
    ids_writer.close()
    pos_writer.close()
    mask_writer.close()
    np.save(stem.with_suffix(".offsets.npy"), np.asarray(offsets, dtype=np.int64))

    meta = {
        "format": "minideepseek-pretrain-v1",
        "seq_len": int(seq_len),
        "packs": int(packs),
        "tokens": int(total),
        "target_tokens": int(total - docs),
        "documents_kept": int(docs),
        "domain": domain,
        "tokenizer_hash": tokenizer.sha256,
        "tokenizer_vocab": tokenizer.vocab_size,
        "sources": [str(Path(p)) for p in inputs],
        "skip_tokens": int(skip_tokens),
        "max_tokens": max_tokens,
        "streaming": True,
    }
    out.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    print(
        f"{out.name}: {total/1e6:.1f}M tokens, {packs} packs, {docs} docs, "
        f"{elapsed/60:.1f} min",
        flush=True,
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--skip-tokens", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--domain", default="general")
    args = parser.parse_args(argv)
    pack(
        [Path(item) for item in args.input],
        Path(args.out),
        seq_len=args.seq_len,
        skip_tokens=args.skip_tokens,
        max_tokens=args.max_tokens,
        domain=args.domain,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
