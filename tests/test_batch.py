from __future__ import annotations

import torch

from minideepseek.model.batch import (
    PackedBatch,
    concat_batches,
    from_padded,
    from_sequences,
    shift_targets,
    valid_token_count,
)


def test_from_sequences_marks_document_starts():
    batch = from_sequences([[1, 2, 3], [4, 5]])
    assert batch.num_tokens == 5
    assert batch.num_documents == 2
    assert batch.positions.tolist() == [0, 1, 2, 0, 1]
    assert batch.target_mask.tolist() == [False, True, True, False, True]
    assert batch.cu_seqlens.tolist() == [0, 3, 5]


def test_shift_targets_respects_document_boundaries():
    batch = from_sequences([[1, 2, 3], [4, 5]])
    targets, valid = shift_targets(batch, 0)
    # logits at 0..3 predict tokens 1..4; the token after doc 0 (index 3) is a
    # document start and must not be predicted from the previous document.
    assert targets.tolist() == [2, 3, 4, 5]
    assert valid.tolist() == [True, True, False, True]
    targets, valid = shift_targets(batch, 1)
    assert valid.tolist() == [True, False, False]
    assert valid_token_count(batch, 0) == 3


def test_concat_batches_offsets():
    first = from_sequences([[1, 2]])
    second = from_sequences([[3, 4], [5]])
    combined = concat_batches([first, second])
    assert combined.num_documents == 3
    assert combined.cu_seqlens.tolist() == [0, 2, 4, 5]
    assert combined.positions.tolist() == [0, 1, 0, 1, 0]


def test_from_padded_removes_padding():
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    batch = from_padded(input_ids, attention_mask=attention)
    assert batch.input_ids.tolist() == [1, 2, 3, 4, 5]
    assert batch.num_documents == 2


def test_batch_validation():
    try:
        PackedBatch(
            input_ids=torch.zeros(3, dtype=torch.long),
            positions=torch.zeros(3, dtype=torch.long),
            target_mask=torch.zeros(3, dtype=torch.bool),
            cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
        )
    except ValueError as error:
        assert "start at 0" in str(error) or "end at T" in str(error)
    else:
        raise AssertionError("expected a ValueError")
