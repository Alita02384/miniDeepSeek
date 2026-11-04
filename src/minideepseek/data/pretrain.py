"""Pretraining dataset: tokenize JSONL documents and pack them into sequences.

The on-disk layout is a flat token buffer plus ``offsets`` that delimit packs of
at most ``seq_len`` tokens.  ``positions`` restart at every document, matching
torchtitan's packed/varlen attention metadata.  No padding is stored.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..model.batch import PackedBatch
from ..tokenizer import MiniDeepSeekTokenizer, load_tokenizer
from .io import default_text, iter_jsonl, normalize_whitespace, save_flat_dataset

DEFAULT_MIN_CHARS = 32


@dataclass
class PackedPretrainDataset:
    """Memory-mapped, packed causal-LM dataset."""

    path: str | Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        archive = np.load(self.path, mmap_mode="r")
        self.input_ids = archive["input_ids"]
        self.positions = archive["positions"]
        self.target_mask = archive["target_mask"]
        self.offsets = archive["offsets"]
        self.meta: dict[str, Any] = {}
        meta_path = self.path.with_suffix(".json")
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text())

    def __len__(self) -> int:
        return int(len(self.offsets) - 1)

    @property
    def num_tokens(self) -> int:
        return int(len(self.input_ids))

    def __getitem__(self, index: int) -> PackedBatch:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        input_ids = torch.from_numpy(np.asarray(self.input_ids[start:end], dtype=np.int64))
        positions = torch.from_numpy(np.asarray(self.positions[start:end], dtype=np.int64))
        mask = torch.from_numpy(np.asarray(self.target_mask[start:end], dtype=bool))
        cu_seqlens = _cu_seqlens_from_positions(positions)
        return PackedBatch(
            input_ids=input_ids, positions=positions, target_mask=mask, cu_seqlens=cu_seqlens
        )


def _cu_seqlens_from_positions(positions: torch.Tensor) -> torch.Tensor:
    if positions.numel() == 0:
        raise ValueError("empty pack")
    starts = (positions == 0).nonzero(as_tuple=False).flatten().to(torch.int32)
    if starts.numel() == 0 or int(starts[0]) != 0:
        raise ValueError("positions must start a document at index 0")
    end = torch.tensor([positions.numel()], dtype=torch.int32)
    return torch.cat([starts, end])


def iter_documents(
    paths: Iterable[str | Path],
    *,
    max_records: int | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> Iterator[str]:
    seen = 0
    for path in paths:
        for record in iter_jsonl(path):
            if max_records is not None and seen >= max_records:
                return
            seen += 1
            try:
                text = default_text(record)
            except ValueError:
                continue
            text = normalize_whitespace(text)
            if len(text) < min_chars:
                continue
            yield text


def tokenize_documents(
    tokenizer: MiniDeepSeekTokenizer,
    texts: Iterable[str],
    *,
    seq_len: int,
    add_bos: bool = True,
    add_eos: bool = True,
    max_tokens: int | None = None,
) -> Iterator[list[int]]:
    """Tokenize one document into one or more chunks of at most ``seq_len`` tokens."""
    budget = 2 if (add_bos and add_eos) else 1
    chunk_size = max(seq_len - budget, 1)
    produced = 0
    for text in texts:
        ids = tokenizer.encode(text)
        if not ids:
            continue
        # Long documents are split into independent chunks; each chunk is a
        # document for attention purposes.
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            tokens = [tokenizer.bos_token_id] if add_bos else []
            tokens.extend(chunk)
            if add_eos:
                tokens.append(tokenizer.eos_token_id)
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            yield tokens
            produced += len(tokens)
            if max_tokens is not None and produced >= max_tokens:
                return


def pack_documents(
    docs: Iterable[list[int]],
    *,
    seq_len: int,
    max_tokens: int | None = None,
    pad_token_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedily pack documents into variable-length packs of <= ``seq_len`` tokens."""
    inputs: list[int] = []
    positions: list[int] = []
    masks: list[bool] = []
    offsets: list[int] = [0]
    current_ids: list[int] = []
    current_pos: list[int] = []
    current_mask: list[bool] = []
    total = 0

    def flush() -> None:
        nonlocal current_ids, current_pos, current_mask
        if not current_ids:
            return
        inputs.extend(current_ids)
        positions.extend(current_pos)
        masks.extend(current_mask)
        offsets.append(len(inputs))
        current_ids, current_pos, current_mask = [], [], []

    for doc in docs:
        if not doc:
            continue
        if len(doc) > seq_len:
            doc = doc[:seq_len]
        if len(current_ids) + len(doc) > seq_len and current_ids:
            flush()
        for offset_in_doc, token in enumerate(doc):
            current_ids.append(int(token))
            current_pos.append(offset_in_doc)
            # The first token of every document has no in-document predecessor.
            current_mask.append(offset_in_doc != 0)
        total += len(doc)
        if max_tokens is not None and total >= max_tokens:
            break
    flush()
    return (
        np.asarray(inputs, dtype=np.int32),
        np.asarray(positions, dtype=np.int32),
        np.asarray(masks, dtype=bool),
        np.asarray(offsets, dtype=np.int64),
    )


def build_pretrain_dataset(
    paths: list[str | Path],
    output_path: str | Path,
    *,
    tokenizer: MiniDeepSeekTokenizer | None = None,
    seq_len: int = 4096,
    max_records: int | None = None,
    max_tokens: int | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    seed: int = 1234,
    dedup: bool = True,
    quality_filter: bool = True,
    domain: str = "general",
) -> dict[str, Any]:
    """Tokenize + pack a pretraining corpus into ``output_path`` (``.npz``)."""
    from .dedup import MinHashIndex
    from .quality import code_license_ok, score_text

    tokenizer = tokenizer or load_tokenizer()
    raw = iter_documents(paths, max_records=max_records, min_chars=min_chars)
    dedup_index = MinHashIndex() if dedup else None
    kept = dropped_quality = dropped_dup = dropped_license = 0

    def filtered_texts():
        nonlocal kept, dropped_quality, dropped_dup, dropped_license
        for text in raw:
            if dedup_index is not None and dedup_index.is_duplicate(text):
                dropped_dup += 1
                continue
            if domain == "code" and not code_license_ok(text):
                dropped_license += 1
                continue
            if quality_filter:
                result = score_text(text, domain=domain)
                if not result.accepted:
                    dropped_quality += 1
                    continue
            kept += 1
            yield text

    docs = tokenize_documents(
        tokenizer, filtered_texts(), seq_len=seq_len, max_tokens=max_tokens
    )
    input_ids, positions, target_mask, offsets = pack_documents(
        docs, seq_len=seq_len, max_tokens=max_tokens, pad_token_id=tokenizer.pad_token_id
    )
    meta = {
        "format": "minideepseek-pretrain-v1",
        "seq_len": int(seq_len),
        "packs": int(len(offsets) - 1),
        "tokens": int(len(input_ids)),
        "target_tokens": int(target_mask.sum()),
        "documents_kept": kept,
        "dropped_quality": dropped_quality,
        "dropped_duplicate": dropped_dup,
        "dropped_license": dropped_license,
        "dedup": bool(dedup),
        "quality_filter": bool(quality_filter),
        "domain": domain,
        "tokenizer_hash": tokenizer.sha256,
        "tokenizer_vocab": tokenizer.vocab_size,
        "sources": [str(Path(p)) for p in paths],
        "max_records": max_records,
        "max_tokens": max_tokens,
        "seed": int(seed),
    }
    save_flat_dataset(
        output_path,
        input_ids=input_ids,
        positions=positions,
        target_mask=target_mask,
        offsets=offsets,
        meta=meta,
    )
    return meta


def iter_batches(
    dataset: PackedPretrainDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    seed: int = 1234,
    shard_rank: int = 0,
    shard_world_size: int = 1,
):
    """Yield ``batch_size`` packs (each pack is already a multi-document batch)."""
    from ..model.batch import concat_batches

    indices = list(range(len(dataset)))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(indices)
    if shard_world_size > 1:
        indices = indices[shard_rank::shard_world_size]
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        yield concat_batches([dataset[index] for index in chunk])
