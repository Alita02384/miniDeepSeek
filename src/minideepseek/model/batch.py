"""Packed, document-aware sequence batches.

miniDeepSeek follows the V4 training data layout: sequences are concatenated
into one flat token stream and document boundaries are carried by ``positions``
(each document starts at position 0).  Attention metadata is derived from those
positions, so two documents in one batch never attend to each other.

``target_mask[t]`` means "token ``t`` is a prediction target".  Logits at
position ``t`` predict token ``t + 1`` (and, for MTP depth ``d``, token
``t + 1 + d``), so the first token of every document is never a target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PackedBatch:
    """A flat, optionally multi-document batch.

    Args:
        input_ids: ``[T]`` token ids.
        positions: ``[T]`` position ids; every document starts at 0.
        target_mask: ``[T]`` bool mask of tokens that should be predicted.
        cu_seqlens: ``[B + 1]`` cumulative document lengths.
        padding_mask: optional ``[T]`` bool mask where ``True`` marks padding.
    """

    input_ids: Tensor
    positions: Tensor
    target_mask: Tensor
    cu_seqlens: Tensor
    padding_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1:
            raise ValueError(f"input_ids must be 1-D, got {tuple(self.input_ids.shape)}")
        if self.positions.shape != self.input_ids.shape:
            raise ValueError("positions and input_ids must have the same shape")
        if self.target_mask.shape != self.input_ids.shape:
            raise ValueError("target_mask and input_ids must have the same shape")
        if self.cu_seqlens.ndim != 1 or self.cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a 1-D tensor with at least 2 entries")
        if int(self.cu_seqlens[0]) != 0 or int(self.cu_seqlens[-1]) != self.input_ids.numel():
            raise ValueError(
                f"cu_seqlens must start at 0 and end at T={self.input_ids.numel()}; "
                f"got {self.cu_seqlens.tolist()[:3]}...{int(self.cu_seqlens[-1])}"
            )
        if self.padding_mask is not None and self.padding_mask.shape != self.input_ids.shape:
            raise ValueError("padding_mask and input_ids must have the same shape")

    @property
    def num_tokens(self) -> int:
        return int(self.input_ids.numel())

    @property
    def num_documents(self) -> int:
        return int(self.cu_seqlens.numel() - 1)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> PackedBatch:
        kwargs = {"non_blocking": non_blocking}
        return PackedBatch(
            input_ids=self.input_ids.to(device, **kwargs),
            positions=self.positions.to(device, **kwargs),
            target_mask=self.target_mask.to(device, **kwargs),
            cu_seqlens=self.cu_seqlens.to(device, **kwargs),
            padding_mask=(
                None if self.padding_mask is None else self.padding_mask.to(device, **kwargs)
            ),
        )

    def doc_ids(self) -> Tensor:
        """Return a ``[T]`` tensor mapping each token to its document index."""
        lengths = torch.diff(self.cu_seqlens).to(self.input_ids.device)
        return torch.repeat_interleave(
            torch.arange(lengths.numel(), device=self.input_ids.device), lengths
        )

    def last_token_indices(self) -> Tensor:
        """Return the index of the last token of every document."""
        return (self.cu_seqlens[1:] - 1).to(self.input_ids.device)


def from_sequences(
    sequences: list[list[int]],
    *,
    target_masks: list[list[bool]] | None = None,
    device: torch.device | str | None = None,
) -> PackedBatch:
    """Pack a list of variable-length sequences into one :class:`PackedBatch`.

    If ``target_masks`` is omitted, every token except each document's first
    token is a target.
    """
    if not sequences:
        raise ValueError("at least one sequence is required")
    lengths = [len(seq) for seq in sequences]
    if any(length == 0 for length in lengths):
        raise ValueError(f"empty sequence in batch: lengths={lengths}")
    if target_masks is not None:
        if len(target_masks) != len(sequences):
            raise ValueError("target_masks must have one entry per sequence")
        if any(len(m) != n for m, n in zip(target_masks, lengths, strict=True)):
            raise ValueError("target_masks must match sequence lengths")

    total = sum(lengths)
    input_ids = torch.empty(total, dtype=torch.long)
    positions = torch.empty(total, dtype=torch.long)
    mask = torch.zeros(total, dtype=torch.bool)
    offset = 0
    for index, length in enumerate(lengths):
        ids = torch.tensor(sequences[index], dtype=torch.long)
        input_ids[offset : offset + length] = ids
        positions[offset : offset + length] = torch.arange(length, dtype=torch.long)
        if target_masks is None:
            if length > 1:
                mask[offset + 1 : offset + length] = True
        else:
            mask[offset : offset + length] = torch.tensor(
                target_masks[index], dtype=torch.bool
            )
        offset += length
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    batch = PackedBatch(
        input_ids=input_ids,
        positions=positions,
        target_mask=mask,
        cu_seqlens=cu_seqlens,
    )
    if device is not None:
        batch = batch.to(device)
    return batch


def from_padded(
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None = None,
    positions: Tensor | None = None,
    target_mask: Tensor | None = None,
) -> PackedBatch:
    """Convert a ``[B, T]`` padded batch into a packed batch.

    Padding tokens are removed.  ``positions`` may be supplied; otherwise each
    row restarts at 0 (one document per row).  ``target_mask``, when supplied,
    must have the same shape as ``input_ids``.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"expected [B, T] input_ids, got {tuple(input_ids.shape)}")
    b, t = input_ids.shape
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    sequences: list[list[int]] = []
    masks: list[list[bool]] | None = [] if (target_mask is not None or positions is not None) else None
    for row in range(b):
        keep = attention_mask[row].bool()
        if not keep.any():
            continue
        ids = input_ids[row][keep].tolist()
        sequences.append(ids)
        if masks is not None:
            if target_mask is not None:
                masks.append(target_mask[row][keep].tolist())
            else:
                # Default: first token of each document is not a target.
                doc_positions = positions[row][keep]
                masks.append([bool(p > 0) for p in doc_positions.tolist()])
    return from_sequences(sequences, target_masks=masks)


def shift_targets(batch: PackedBatch, shift: int) -> tuple[Tensor, Tensor]:
    """Return targets and validity mask for MTP depth ``shift``.

    For ``shift=0`` the target at position ``t`` is ``input_ids[t + 1]``.  For
    ``shift=d`` it is ``input_ids[t + 1 + d]``, which is what MTP depth ``d``
    predicts.  Tokens are only valid when both ends belong to the same document.
    """
    if shift < 0:
        raise ValueError("shift must be non-negative")
    total = batch.num_tokens
    step = shift + 1
    if step >= total:
        return batch.input_ids.new_empty(0), batch.target_mask.new_empty(0)
    logits_slice = slice(0, total - step)
    target_positions = batch.positions[step:]
    source_positions = batch.positions[logits_slice]
    valid = target_positions == source_positions + step
    valid = valid & batch.target_mask[step:]
    if batch.padding_mask is not None:
        valid = valid & ~batch.padding_mask[step:] & ~batch.padding_mask[logits_slice]
    return batch.input_ids[step:], valid


def valid_token_count(batch: PackedBatch, shift: int) -> int:
    """Number of valid prediction tokens for MTP depth ``shift``."""
    _, mask = shift_targets(batch, shift)
    return int(mask.sum())


def concat_batches(batches: list[PackedBatch]) -> PackedBatch:
    """Concatenate batches into one multi-document batch."""
    if not batches:
        raise ValueError("no batches to concatenate")
    input_ids = torch.cat([b.input_ids for b in batches])
    positions = torch.cat([b.positions for b in batches])
    target_mask = torch.cat([b.target_mask for b in batches])
    offsets = []
    running = 0
    for b in batches:
        offsets.extend((b.cu_seqlens[:-1] + running).tolist())
        running += b.num_tokens
    offsets.append(running)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32)
    return PackedBatch(
        input_ids=input_ids,
        positions=positions,
        target_mask=target_mask,
        cu_seqlens=cu_seqlens,
    )
