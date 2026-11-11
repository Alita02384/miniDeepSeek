"""Epoch prompt sampler: no replacement inside an epoch."""

from __future__ import annotations

import pytest

from minideepseek.rl.sampling import EpochPromptSampler


def test_take_is_without_replacement_inside_an_epoch():
    sampler = EpochPromptSampler(10, seed=7)
    first = sampler.take(3)
    second = sampler.take(7)
    assert len(first) == 3
    assert len(second) == 7
    assert sorted(first + second) == list(range(10))
    assert sampler.epochs_started == 1


def test_take_spans_epoch_boundary_and_reshuffles():
    # Consume three of four prompts, then cross into epoch two in one call.
    sampler = EpochPromptSampler(4, seed=1)
    sampler.take(3)
    crossing = sampler.take(3)
    assert len(crossing) == 3
    assert sampler.epochs_started == 2


def test_take_caps_at_pool_size():
    sampler = EpochPromptSampler(5, seed=1)
    indices = sampler.take(99)
    assert sorted(indices) == list(range(5))


def test_same_seed_gives_same_order():
    left = EpochPromptSampler(20, seed=1234)
    right = EpochPromptSampler(20, seed=1234)
    assert left.take(7) == right.take(7)
    assert left.take(7) == right.take(7)


def test_empty_pool_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        EpochPromptSampler(0)
