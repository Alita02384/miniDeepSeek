"""Weighted multi-source batch mixing.

A pre-training corpus is usually a mixture of domains with target token
fractions (roadmap section 4.2).  ``WeightedBatchMixer`` turns several batch
iterators into one deterministic interleaving: the number of batches assigned
to each source is proportional to its weight, using the largest-remainder rule,
then the schedule is shuffled with a fixed seed.  Sources restart when their
iterator is exhausted, so the caller can drive a fixed number of optimizer
steps.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from ..model.batch import PackedBatch

__all__ = ["BatchSource", "WeightedBatchMixer"]


@dataclass
class BatchSource:
    name: str
    factory: Callable[[], Iterator[PackedBatch]]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("source weight must be non-negative")


def _quotas(weights: list[float], total: int) -> list[int]:
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("at least one source needs a positive weight")
    if total <= 0:
        raise ValueError("total_batches must be positive")
    raw = [weight / total_weight * total for weight in weights]
    floors = [int(math.floor(value)) for value in raw]
    remainder = total - sum(floors)
    order = sorted(
        range(len(weights)),
        key=lambda index: (raw[index] - floors[index], weights[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        floors[index] += 1
    return floors


class _MixerIterator:
    """Iterator view over a mixer that also exposes the active source."""

    def __init__(self, mixer: WeightedBatchMixer) -> None:
        self._mixer = mixer

    def __iter__(self) -> _MixerIterator:
        return self

    def __next__(self) -> PackedBatch:
        return self._mixer.next_batch()

    @property
    def current_index(self) -> int | None:
        return self._mixer.current_index

    @property
    def current_source(self) -> str | None:
        index = self._mixer.current_index
        return None if index is None else self._mixer.sources[index].name


class WeightedBatchMixer:
    """Yield batches from multiple sources with fixed proportions.

    ``lockstep=True`` keeps every optimizer step domain-homogeneous: the
    schedule is built over *steps*, and each slot yields ``batches_per_step``
    consecutive batches from the same source.  With ``lockstep=False`` the
    schedule is built over batches and sources interleave at batch granularity.

    Sources restart when their iterator is exhausted, so the caller can drive a
    fixed number of optimizer steps.
    """

    def __init__(
        self,
        sources: list[BatchSource],
        *,
        total_batches: int,
        seed: int = 1234,
        shuffle: bool = True,
        batches_per_step: int = 1,
        lockstep: bool = False,
    ):
        if not sources:
            raise ValueError("at least one source is required")
        if batches_per_step < 1:
            raise ValueError("batches_per_step must be positive")
        self.sources = sources
        self.lockstep = bool(lockstep)
        self.batches_per_step = int(batches_per_step) if self.lockstep else 1
        # ``lockstep`` schedules one source per optimizer step; otherwise one
        # source per micro-batch.
        slots = -(-int(total_batches) // self.batches_per_step)
        quotas = _quotas([source.weight for source in sources], slots)
        schedule = [
            index for index, count in enumerate(quotas) for _ in range(count)
        ]
        if shuffle:
            random.Random(seed).shuffle(schedule)
        self.schedule = schedule
        self._iterators: dict[int, Iterator[PackedBatch]] = {}
        self._restarts = {index: 0 for index in range(len(sources))}
        self._cursor = 0
        self._used = 0
        self._current: int | None = None

    @property
    def current_index(self) -> int | None:
        """Index of the source that produced the most recent batch."""
        return self._current

    def _next(self, index: int) -> PackedBatch:
        if index not in self._iterators:
            self._iterators[index] = iter(self.sources[index].factory())
        try:
            return next(self._iterators[index])
        except StopIteration:
            self._restarts[index] += 1
            self._iterators[index] = iter(self.sources[index].factory())
            return next(self._iterators[index])

    def next_batch(self) -> PackedBatch:
        """Return the next batch, advancing the source schedule as needed."""
        if self._cursor >= len(self.schedule):
            raise StopIteration
        index = self.schedule[self._cursor]
        self._current = index
        batch = self._next(index)
        self._used += 1
        if self._used >= self.batches_per_step:
            self._used = 0
            self._cursor += 1
        return batch

    def __iter__(self) -> Iterator[PackedBatch]:
        return _MixerIterator(self)

    def __len__(self) -> int:
        return len(self.schedule) * self.batches_per_step

    def restart_counts(self) -> dict[str, int]:
        return {
            source.name: self._restarts[index]
            for index, source in enumerate(self.sources)
        }
