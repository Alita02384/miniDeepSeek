"""Prompt sampling for GRPO / OPD.

The trainers used to reshuffle the whole prompt pool at every step and take the
first ``count`` entries.  That is sampling *with replacement*: 1,600 code GRPO
draws from 3,135 prompts hit ~1,251 distinct prompts, and 1,600 OPD draws from
9,135 prompts hit ~1,467.  Repeats inside one step and across adjacent steps
waste capacity, and there is no epoch boundary.

``EpochPromptSampler`` walks a shuffled permutation and only reshuffles after
every prompt has been used once.  For the shipped budgets it raises distinct
prompt coverage to the full draw count (e.g. code 1,251 -> 1,600, +28%) and
never repeats a prompt before the epoch ends.
"""

from __future__ import annotations

import random

__all__ = ["EpochPromptSampler"]


class EpochPromptSampler:
    """Draw fixed-size prompt batches without replacement inside an epoch."""

    def __init__(self, size: int, *, seed: int = 1234):
        if size < 1:
            raise ValueError("prompt pool must not be empty")
        self.size = int(size)
        self.rng = random.Random(int(seed))
        self._order: list[int] = []
        self._cursor = 0
        self.epochs_started = 0

    def take(self, count: int) -> list[int]:
        """Return up to ``count`` indices, reshuffling only at epoch boundaries."""
        count = min(max(int(count), 1), self.size)
        indices: list[int] = []
        while len(indices) < count:
            if self._cursor >= len(self._order):
                order = list(range(self.size))
                self.rng.shuffle(order)
                self._order = order
                self._cursor = 0
                self.epochs_started += 1
            take = min(count - len(indices), len(self._order) - self._cursor)
            indices.extend(self._order[self._cursor : self._cursor + take])
            self._cursor += take
        return indices
