"""Supervised fine-tuning dataset with response-only loss masks.

Accepted input schemas:

* MiniMind: ``{"conversations": [{"role", "content", "reasoning_content"}, ...]}``
* roadmap:  ``{"messages": [...], "domain", "source", "license", ...}``

The loss mask is token-pattern based (see ``tokenizer.assistant_token_mask``),
so no string-span bookkeeping is needed and multi-turn/tool data works.
"""

from __future__ import annotations

import json
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
    # ``array``/``bytearray`` keep a 600M-token SFT corpus at ~6 GB instead of
    # the ~50 GB a list-of-Python-ints would need.  This mirrors the pretrain
    # packing buffers and produces byte-identical arrays.
    flat_ids = array("i")
    flat_positions = array("i")
    flat_mask = bytearray()
    offsets = array("q", [0])
    used = 0
    skipped = 0
    for path in paths:
        for record in iter_jsonl(path):
            if max_records is not None and used >= max_records:
                break
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
