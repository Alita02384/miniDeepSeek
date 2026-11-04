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


class WeightedBatchMixer:
    """Yield batches from multiple sources with fixed batch proportions."""

    def __init__(
        self,
        sources: list[BatchSource],
        *,
        total_batches: int,
        seed: int = 1234,
        shuffle: bool = True,
    ):
        if not sources:
            raise ValueError("at least one source is required")
        self.sources = sources
        quotas = _quotas([source.weight for source in sources], total_batches)
        schedule = [
            index for index, count in enumerate(quotas) for _ in range(count)
        ]
        if shuffle:
            random.Random(seed).shuffle(schedule)
        self.schedule = schedule
        self._iterators: dict[int, Iterator[PackedBatch]] = {}
        self._started: set[int] = set()
        self._restarts = {index: 0 for index in range(len(sources))}

    def _next(self, index: int) -> PackedBatch:
        if index not in self._iterators:
            self._iterators[index] = iter(self.sources[index].factory())
            self._started.add(index)
        try:
            return next(self._iterators[index])
        except StopIteration:
            self._restarts[index] += 1
            self._iterators[index] = iter(self.sources[index].factory())
            return next(self._iterators[index])

    def __iter__(self) -> Iterator[PackedBatch]:
        for index in self.schedule:
            yield self._next(index)

    def __len__(self) -> int:
        return len(self.schedule)

    def restart_counts(self) -> dict[str, int]:
        return {
            source.name: self._restarts[index]
            for index, source in enumerate(self.sources)
        }
