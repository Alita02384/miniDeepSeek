"""Supervised fine-tuning dataset with response-only loss masks.

Accepted input schemas:

* MiniMind: ``{"conversations": [{"role", "content", "reasoning_content"}, ...]}``
* roadmap:  ``{"messages": [...], "domain", "source", "license", ...}``

The loss mask is token-pattern based (see ``tokenizer.assistant_token_mask``),
so no string-span bookkeeping is needed and multi-turn/tool data works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..model.batch import PackedBatch, concat_batches
from ..tokenizer import MiniDeepSeekTokenizer, load_tokenizer, normalize_conversation
from .io import iter_jsonl, save_flat_dataset


@dataclass
class SFTDataset:
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

    @property
    def num_target_tokens(self) -> int:
        return int(self.target_mask.sum())

    def __getitem__(self, index: int) -> PackedBatch:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        input_ids = torch.from_numpy(np.asarray(self.input_ids[start:end], dtype=np.int64))
        positions = torch.arange(end - start, dtype=torch.long)
        mask = torch.from_numpy(np.asarray(self.target_mask[start:end], dtype=bool))
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
    encoded = tokenizer.encode_chat(
        messages,
        add_generation_prompt=add_generation_prompt,
        tools=tools if isinstance(tools, list) else None,
        max_length=seq_len,
    )
    if sum(encoded.target_mask) < 2:
        return None
    return encoded.input_ids, encoded.target_mask


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
) -> dict[str, Any]:
    tokenizer = tokenizer or load_tokenizer()
    flat_ids: list[int] = []
    flat_positions: list[int] = []
    flat_mask: list[bool] = []
    offsets: list[int] = [0]
    used = 0
    skipped = 0
    for path in paths:
        for record in iter_jsonl(path):
            if max_records is not None and used >= max_records:
                break
            encoded = encode_sft_record(
                record,
                tokenizer,
                seq_len=seq_len,
                add_generation_prompt=add_generation_prompt,
            )
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
                break
        if max_records is not None and used >= max_records:
            break
        if max_tokens is not None and len(flat_ids) >= max_tokens:
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
        "sources": [str(Path(p)) for p in paths],
        "add_generation_prompt": bool(add_generation_prompt),
        "seed": int(seed),
    }
    save_flat_dataset(
        output_path,
        input_ids=np.asarray(flat_ids, dtype=np.int32),
        positions=np.asarray(flat_positions, dtype=np.int32),
        target_mask=np.asarray(flat_mask, dtype=bool),
        offsets=np.asarray(offsets, dtype=np.int64),
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
):
    """Yield multi-document batches greedily filled up to ``batch_tokens``."""
    order = list(range(len(dataset)))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(order)
    if shard_world_size > 1:
        order = order[shard_rank::shard_world_size]
    group: list[PackedBatch] = []
    total = 0
    for index in order:
        sample = dataset[index]
        if group and total + sample.num_tokens > batch_tokens:
            yield concat_batches(group)
            group, total = [], 0
        group.append(sample)
        total += sample.num_tokens
    if group:
        yield concat_batches(group)
