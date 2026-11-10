# Copyright (c) 2026 miniDeepSeek contributors.
"""CUDA-graph wrapper for the one-token decode step.

The eager decode step launches ~1k tiny kernels (7 layers x mHC sinkhorn x
MoE x attention).  At batch 64 the GPU finishes in a few milliseconds while the
host spends 50+ ms launching, so GRPO rollout is launch-bound.  A CUDA graph
replays the whole step from one launch and removes that overhead.

The graph is captured for a fixed batch size (bucketed powers of two).  Every
input tensor is static and copied into before ``replay``; the cache tensors are
mutated in place, which graphs support.  Padding rows point at one dedicated
scratch slot, so capturing a bucket never touches a live sequence.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

DEFAULT_BUCKETS: tuple[int, ...] = (
    1,
    2,
    4,
    8,
    16,
    32,
    48,
    64,
    96,
    128,
    160,
    192,
    256,
    320,
    384,
    512,
)


class GraphDecodeRunner:
    """Bucketed CUDA-graph capture around ``model.decode``."""

    def __init__(
        self,
        model,
        cache,
        *,
        buckets: Sequence[int] = DEFAULT_BUCKETS,
        warmup_steps: int = 2,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require a CUDA device")
        self.model = model
        self.cache = cache
        self.buckets = tuple(sorted({int(b) for b in buckets if int(b) > 0}))
        self.warmup_steps = int(warmup_steps)
        # One extra row is reserved for padding; the caller must keep live
        # slots below ``cache.num_slots - 1``.
        self.scratch_slot = int(cache.num_slots) - 1
        self._graphs: dict[int, tuple] = {}
        self.disabled = False

    @property
    def capacity(self) -> int:
        return int(self.cache.num_slots) - 1

    def _bucket_for(self, batch: int) -> int | None:
        for bucket in self.buckets:
            if bucket >= batch:
                return bucket
        return None

    def _capture(self, bucket: int) -> None:
        device = self.cache.device
        tokens = torch.zeros(bucket, dtype=torch.long, device=device)
        positions = torch.zeros(bucket, dtype=torch.long, device=device)
        slots = torch.full(
            (bucket,), self.scratch_slot, dtype=torch.long, device=device
        )
        # Warm up on the scratch slot only: live cache rows stay untouched.
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                self.model.decode_step(tokens, positions, self.cache, slots)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.model.decode_step(tokens, positions, self.cache, slots)
        self._graphs[bucket] = (graph, tokens, positions, slots, output)

    def step(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(tokens.numel())
        bucket = self._bucket_for(batch)
        if bucket is None:
            return self.model.decode_step(tokens, positions, self.cache, slots)
        if bucket not in self._graphs:
            if self.disabled:
                return self.model.decode_step(tokens, positions, self.cache, slots)
            try:
                self._capture(bucket)
            except Exception as error:  # pragma: no cover - depends on the GPU
                import warnings

                warnings.warn(
                    f"CUDA graph capture failed for batch {bucket}, "
                    f"falling back to eager decode: {error}",
                    stacklevel=2,
                )
                self.disabled = True
                return self.model.decode_step(tokens, positions, self.cache, slots)
        graph, static_tokens, static_positions, static_slots, output = self._graphs[bucket]
        static_tokens[:batch].copy_(tokens)
        static_positions[:batch].copy_(positions)
        static_slots[:batch].copy_(slots)
        if bucket > batch:
            static_tokens[batch:] = 0
            static_positions[batch:] = 0
            static_slots[batch:] = self.scratch_slot
        graph.replay()
        return output[:batch]
