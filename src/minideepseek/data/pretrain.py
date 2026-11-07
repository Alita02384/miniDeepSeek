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
from .io import default_text, iter_jsonl, load_flat_dataset, normalize_whitespace, save_flat_dataset

DEFAULT_MIN_CHARS = 32


@dataclass
class PackedPretrainDataset:
    """Memory-mapped, packed causal-LM dataset."""

    path: str | Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.input_ids, self.positions, self.target_mask, self.offsets = (
            load_flat_dataset(self.path)
        )
        self.meta: dict[str, Any] = {}
        meta_path = self.path.with_suffix(".json")
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return int(len(self.offsets) - 1)

    @property
    def num_tokens(self) -> int:
        return int(len(self.input_ids))

    def __getitem__(self, index: int) -> PackedBatch:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        input_ids = torch.from_numpy(
            np.array(self.input_ids[start:end], dtype=np.int64, copy=True)
        )
        positions = torch.from_numpy(
            np.array(self.positions[start:end], dtype=np.int64, copy=True)
        )
        mask = torch.from_numpy(
            np.array(self.target_mask[start:end], dtype=bool, copy=True)
        )
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
    paragraph_dedup: bool = True,
    domain: str = "general",
) -> dict[str, Any]:
    """Tokenize + pack a pretraining corpus into ``output_path`` (``.npz``)."""
    from .dedup import MinHashIndex, ParagraphDeduper
    from .quality import code_license_ok, score_text

    tokenizer = tokenizer or load_tokenizer()
    raw = iter_documents(paths, max_records=max_records, min_chars=min_chars)
    dedup_index = MinHashIndex() if dedup else None
    paragraph_deduper = ParagraphDeduper() if paragraph_dedup else None
    kept = dropped_quality = dropped_dup = dropped_license = 0

    def filtered_texts():
        nonlocal kept, dropped_quality, dropped_dup, dropped_license
        for text in raw:
            if dedup_index is not None and dedup_index.is_duplicate(text):
                dropped_dup += 1
                continue
            if paragraph_deduper is not None:
                text = paragraph_deduper(text)
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
        "paragraph_dedup": bool(paragraph_dedup),
        "paragraphs_removed": int(paragraph_deduper.removed) if paragraph_deduper else 0,
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


def token_schedule(
    base: int,
    final: int | None,
    warmup_ratio: float,
    total_steps: int,
    batch_index: int,
) -> int:
    """Linear batch-token warmup over the first ``warmup_ratio`` of steps."""
    if final is None or final <= 0 or warmup_ratio <= 0:
        return max(int(base), 1)
    if final < base:
        raise ValueError("batch_tokens_final must be >= batch_tokens")
    warmup_steps = max(int(round(total_steps * warmup_ratio)), 1)
    progress = min(max(batch_index / warmup_steps, 0.0), 1.0)
    return max(int(round(base + (final - base) * progress)), 1)


@dataclass
class BatchTokenSchedule:
    """Stateful batch-token warmup shared across loader restarts.

    A dataloader factory is re-created whenever an epoch ends or a run resumes.
    Keeping the warmup counter in the caller-owned schedule object makes the
    budget continue from the global batch index instead of restarting at
    ``base`` every time.
    """

    base: int
    final: int | None
    warmup_ratio: float
    total_steps: int
    index: int = 0

    def next(self) -> int:
        value = token_schedule(
            self.base,
            self.final,
            self.warmup_ratio,
            self.total_steps,
            self.index,
        )
        self.index += 1
        return value


def iter_batches(
    dataset: PackedPretrainDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    seed: int = 1234,
    shard_rank: int = 0,
    shard_world_size: int = 1,
    batch_tokens: int | None = None,
    batch_tokens_final: int | None = None,
    batch_warmup_ratio: float = 0.0,
    total_steps: int = 1,
    seq_len: int = 1,
    schedule: BatchTokenSchedule | None = None,
):
    """Yield packs (each pack is already a multi-document batch).

    When ``batch_tokens`` is given, the pack size is derived per batch from
    :func:`token_schedule` instead of the fixed ``batch_size``.  Pass a shared
    ``schedule`` to keep the warmup progress across loader restarts.
    """
    from ..model.batch import concat_batches

    indices = list(range(len(dataset)))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(indices)
    if shard_world_size > 1:
        indices = indices[shard_rank::shard_world_size]
    index = 0
    batch_index = 0

    def next_budget() -> int | None:
        if batch_tokens is None:
            return None
        if schedule is not None:
            return schedule.next()
        return token_schedule(
            batch_tokens,
            batch_tokens_final,
            batch_warmup_ratio,
            total_steps,
            batch_index,
        )

    while index < len(indices):
        current = next_budget()
        if current is None:
            current = max(int(batch_size), 1) * max(int(seq_len), 1)
        size = max(1, current // max(int(seq_len), 1))
        chunk = indices[index : index + size]
        index += size
        batch_index += 1
        yield concat_batches([dataset[item] for item in chunk])
