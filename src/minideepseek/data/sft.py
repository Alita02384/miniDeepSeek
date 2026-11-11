"""Supervised fine-tuning dataset with response-only loss masks.

Accepted input schemas:

* MiniMind: ``{"conversations": [{"role", "content", "reasoning_content"}, ...]}``
* roadmap:  ``{"messages": [...], "domain", "source", "license", ...}``

The loss mask is token-pattern based (see ``tokenizer.assistant_token_mask``),
so no string-span bookkeeping is needed and multi-turn/tool data works.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pretrain import BatchTokenSchedule

import numpy as np
import torch

from ..model.batch import PackedBatch, concat_batches
from ..tokenizer import MiniDeepSeekTokenizer, load_tokenizer, normalize_conversation
from .io import iter_jsonl, load_flat_dataset, save_flat_dataset


@dataclass
class SFTDataset:
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

    @property
    def num_target_tokens(self) -> int:
        return int(self.target_mask.sum())

    def __getitem__(self, index: int) -> PackedBatch:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        input_ids = torch.from_numpy(
            np.array(self.input_ids[start:end], dtype=np.int64, copy=True)
        )
        positions = torch.arange(end - start, dtype=torch.long)
        mask = torch.from_numpy(
            np.array(self.target_mask[start:end], dtype=bool, copy=True)
        )
        cu_seqlens = torch.tensor([0, end - start], dtype=torch.int32)
        return PackedBatch(
            input_ids=input_ids, positions=positions, target_mask=mask, cu_seqlens=cu_seqlens
        )


def encode_sft_record(
    record: dict[str, Any],
    tokenizer: MiniDeepSeekTokenizer,
    *,
    seq_len: int,
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[bool]] | None:
    messages = normalize_conversation(record)
    tools = record.get("tools")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            tools = None
    if not isinstance(tools, list):
        # MiniMind stores the tool schema on the system message, not at the
        # record top level.
        tools = next(
            (
                message["tools"]
                for message in messages
                if message.get("role") == "system" and message.get("tools")
            ),
            None,
        )
    encoded = tokenizer.encode_chat(
        messages,
        add_generation_prompt=add_generation_prompt,
        tools=tools if isinstance(tools, list) else None,
        max_length=seq_len,
    )
    if sum(encoded.target_mask) < 2:
        return None
    return encoded.input_ids, encoded.target_mask


def _conversation_key(record: dict[str, Any]) -> str:
    """Stable digest of the normalized conversation for exact dedup."""
    messages = normalize_conversation(record)
    payload = json.dumps(
        messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _allocate(counts: list[int], total: int) -> list[int]:
    """Largest-remainder allocation, capped by each source's unique count."""
    if total <= 0:
        return [0] * len(counts)
    available = sum(counts)
    if total >= available:
        return list(counts)
    raw = [count / available * total for count in counts]
    quotas = [int(math.floor(value)) for value in raw]
    remainder = total - sum(quotas)
    order = sorted(
        range(len(counts)),
        key=lambda index: (raw[index] - quotas[index], counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        if quotas[index] < counts[index]:
            quotas[index] += 1
    # A cap from largest-remainder can leave a slot unused when the top
    # candidates are already at their cap; fill from the largest unused source.
    while sum(quotas) < total:
        index = max(
            (i for i in range(len(counts)) if quotas[i] < counts[i]),
            key=lambda i: counts[i] - quotas[i],
            default=None,
        )
        if index is None:
            break
        quotas[index] += 1
    return quotas


def build_sft_dataset(
    paths: list[str | Path],
    output_path: str | Path,
    *,
    tokenizer: MiniDeepSeekTokenizer | None = None,
    seq_len: int = 4096,
    max_records: int | None = None,
    max_tokens: int | None = None,
    add_generation_prompt: bool = False,
    seed: int = 1234,
    shuffle: bool = True,
    dedup: bool = True,
) -> dict[str, Any]:
    """Tokenize SFT sources with sampling, dedup and no file-prefix bias.

    The previous implementation walked each file from the top and stopped at
    ``max_records``; ``seed`` was only written into the meta.  That trained the
    general specialist on an 86%-multi-turn prefix block and the code
    specialist on two of its five sources.  This version scans all records,
    deduplicates exact conversations, allocates the record budget across files
    proportionally to their unique counts, shuffles the selected line indices
    with ``seed``, and only then encodes.  The selection digest in the meta
    makes the exact sample reproducible.
    """
    tokenizer = tokenizer or load_tokenizer()
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("at least one SFT source path is required")

    # Pass 1: index every record by line number, dedup by conversation, and
    # count what each source can contribute.  No tokenization yet, so this is
    # cheap relative to encoding.
    candidates: list[list[int]] = [[] for _ in paths]
    seen: set[str] = set()
    source_stats: list[dict[str, Any]] = []
    for path_index, path in enumerate(paths):
        total = duplicates = invalid = 0
        for line_index, record in enumerate(iter_jsonl(path)):
            total += 1
            try:
                key = _conversation_key(record)
            except ValueError:
                invalid += 1
                continue
            if dedup:
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
            candidates[path_index].append(line_index)
        source_stats.append(
            {
                "path": str(path),
                "total": total,
                "unique": len(candidates[path_index]),
                "duplicates": duplicates,
                "invalid": invalid,
                "selected": 0,
            }
        )

    unique_total = sum(len(items) for items in candidates)
    if unique_total == 0:
        raise ValueError("no usable SFT records were produced")
    budget = unique_total if max_records is None else min(int(max_records), unique_total)
    quotas = _allocate([len(items) for items in candidates], budget)
    selected: list[list[int]] = []
    for path_index, items in enumerate(candidates):
        rng = random.Random(int(seed) + path_index)
        order = list(items)
        if shuffle:
            rng.shuffle(order)
        chosen = sorted(order[: quotas[path_index]])
        selected.append(chosen)
        source_stats[path_index]["selected"] = len(chosen)
        source_stats[path_index]["sampled"] = shuffle

    selection_digest = hashlib.sha256(
        "\n".join(
            f"{path_index}:{line_index}"
            for path_index, lines in enumerate(selected)
            for line_index in lines
        ).encode("utf-8")
    ).hexdigest()

    # Pass 2: stream each file once and encode the selected lines in order.
    flat_ids = array("i")
    flat_positions = array("i")
    flat_mask = bytearray()
    offsets = array("q", [0])
    used = 0
    skipped = 0
    stop = False
    for path_index, path in enumerate(paths):
        wanted = selected[path_index]
        cursor = 0
        if not wanted or stop:
            continue
        for line_index, record in enumerate(iter_jsonl(path)):
            if cursor >= len(wanted):
                break
            if line_index != wanted[cursor]:
                continue
            cursor += 1
            try:
                encoded = encode_sft_record(
                    record,
                    tokenizer,
                    seq_len=seq_len,
                    add_generation_prompt=add_generation_prompt,
                )
            except ValueError:
                skipped += 1
                continue
            if encoded is None:
                skipped += 1
                continue
            ids, mask = encoded
            flat_ids.extend(ids)
            flat_positions.extend(range(len(ids)))
            flat_mask.extend(mask)
            offsets.append(len(flat_ids))
            used += 1
            if max_tokens is not None and len(flat_ids) >= max_tokens:
                stop = True
                break

    if used == 0:
        raise ValueError("no usable SFT records were produced")
    meta = {
        "format": "minideepseek-sft-v1",
        "seq_len": int(seq_len),
        "examples": int(used),
        "skipped": int(skipped),
        "tokens": int(len(flat_ids)),
        "target_tokens": int(sum(flat_mask)),
        "tokenizer_hash": tokenizer.sha256,
        "sources": [str(path) for path in paths],
        "add_generation_prompt": bool(add_generation_prompt),
        "seed": int(seed),
        "shuffle": bool(shuffle),
        "dedup": bool(dedup),
        "max_records": None if max_records is None else int(max_records),
        "max_tokens": None if max_tokens is None else int(max_tokens),
        "unique_records": int(unique_total),
        "selected_records": int(sum(len(lines) for lines in selected)),
        "selection_sha256": selection_digest,
        "source_stats": source_stats,
    }
    if flat_ids.itemsize != 4:  # pragma: no cover - non-32-bit-int platforms
        raise RuntimeError(f"array('i') has itemsize {flat_ids.itemsize}, expected 4")
    save_flat_dataset(
        output_path,
        input_ids=np.frombuffer(flat_ids, dtype=np.int32),
        positions=np.frombuffer(flat_positions, dtype=np.int32),
        target_mask=np.frombuffer(flat_mask, dtype=np.uint8).view(bool),
        offsets=np.frombuffer(offsets, dtype=np.int64),
        meta=meta,
    )
    return meta


def iter_batches(
    dataset: SFTDataset,
    *,
    batch_tokens: int = 4096,
    shuffle: bool = True,
    seed: int = 1234,
    shard_rank: int = 0,
    shard_world_size: int = 1,
    batch_tokens_final: int | None = None,
    batch_warmup_ratio: float = 0.0,
    total_steps: int = 1,
    schedule: BatchTokenSchedule | None = None,
):
    """Yield multi-document batches greedily filled up to the current budget.

    Pass a shared ``BatchTokenSchedule`` to keep warmup progress across loader
    restarts (see ``data.pretrain.iter_batches``).
    """
    from .pretrain import token_schedule

    order = list(range(len(dataset)))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(order)
    if shard_world_size > 1:
        order = order[shard_rank::shard_world_size]
    group: list[PackedBatch] = []
    total = 0
    batch_index = 0
    limit: int | None = None

    def next_limit() -> int:
        if schedule is not None:
            return schedule.next()
        return token_schedule(
            batch_tokens,
            batch_tokens_final,
            batch_warmup_ratio,
            total_steps,
            batch_index,
        )

    for index in order:
        if limit is None:
            limit = next_limit()
        sample = dataset[index]
        if group and total + sample.num_tokens > limit:
            yield concat_batches(group)
            group, total = [], 0
            batch_index += 1
            limit = next_limit()
        group.append(sample)
        total += sample.num_tokens
    if group:
        yield concat_batches(group)
