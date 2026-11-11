# Copyright (c) 2026 miniDeepSeek contributors.
"""Batched KV-cache rollout engine (the production rollout path).

The torch engine in :mod:`minideepseek.rl.rollout` recomputes the whole
sequence at every decode step.  That is the right reference for the
consistency gate, but it makes GRPO/OPD rollout-bound: a 1k-token response
costs ``O(T^2)`` attention.  This engine keeps a slot-addressed KV cache
(:class:`minideepseek.model.cache.RolloutCache`) and decodes one token per
step, which is the same asymptotic complexity as vLLM.

Why not vLLM itself?  vLLM 0.33 has a DeepSeek-V4 implementation, but its
attention backends require SM90 (FlashMLA; the kernel hardcodes
``head_dim_v=512`` and FP8 KV) or SM100/120 (FlashInfer), and it expects the
official 576/512-dim MLA.  The mini-145M model is ``head_dim=64`` bf16 on
SM89, so no vLLM backend can serve it.  This engine implements the same
arithmetic as the training forward, with the compressor/indexer state machines
ported from the official inference implementation, and keeps a
``rollout_backend`` seam so a vLLM backend can be added when its kernels
support the model.

Batch layout: one slot per rollout sample.  The prompt is prefilled once and
forked into the group samples (``RolloutCache.copy_slot``), so group_size
rollouts share the prefill cost.  Decoding compacts the active slots each
step; a finished sequence stops consuming compute.
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from contextlib import contextmanager

import torch

from ..model.cache import RolloutCache
from ..model.wrapper import _sample_tokens
from .graph_decode import GraphDecodeRunner
from .rollout import RolloutGroup, RolloutSample, TorchRolloutEngine


@contextmanager
def _reference_attention():
    """Force the torch gather-attention reference around rollout prefill.

    ``attn_gym`` specializes its Triton kernel on every (top-k, layout) pair;
    rollout prompts have many lengths, so the fused path would recompile for
    each one.  The reference is exact and fast enough for prefill; the decode
    path does not use it at all.
    """
    previous = os.environ.get("MINIDEEPSEEK_ATTN_BACKEND")
    os.environ["MINIDEEPSEEK_ATTN_BACKEND"] = "torch"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MINIDEEPSEEK_ATTN_BACKEND", None)
        else:
            os.environ["MINIDEEPSEEK_ATTN_BACKEND"] = previous


class CachedRolloutEngine(TorchRolloutEngine):
    """KV-cache rollout with batched prefill, group forking and compaction."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        thinking_ratio: float = 1.0,
        max_slots: int = 256,
        max_context: int | None = None,
        use_cuda_graph: bool | None = None,
    ):
        super().__init__(model, tokenizer, thinking_ratio=thinking_ratio)
        self.max_slots = int(max_slots)
        self.max_context = int(max_context or model.max_context_length)
        self._cache: RolloutCache | None = None
        self._graph: GraphDecodeRunner | None = None
        self._cache_generation = 0
        if use_cuda_graph is None:
            use_cuda_graph = bool(
                torch.cuda.is_available() and model.device.type == "cuda"
            )
        self.use_cuda_graph = bool(use_cuda_graph)

    # -- cache reuse ---------------------------------------------------------
    def _ensure_cache(self, slots: int) -> RolloutCache:
        # One extra row is the graph-padding scratch slot.
        capacity = max(slots + 1, 1)
        if self._cache is None or self._cache.num_slots < capacity:
            capacity = max(capacity, self.max_slots + 1)
            self._cache = RolloutCache.from_model(
                self.model,
                num_slots=capacity,
                max_context=self.max_context,
            )
            self._graph = None
        cache = self._cache
        cache.reset(torch.arange(cache.num_slots, device=self.model.device))
        return cache

    @property
    def cache(self) -> RolloutCache | None:
        return self._cache

    def sync_weights(self, state_dict) -> int:
        """No-op: the decode cache reads the training weights directly."""
        del state_dict
        self._cache_generation += 1
        return self._cache_generation

    def check_generator_consistency(self, samples, *, temperature=None) -> float | None:
        """The rollout forward and the actor forward share the same weights."""
        del samples, temperature
        return None

    # -- rollout -------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        records: Sequence,
        *,
        group_size: int = 4,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> list[RolloutGroup]:
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        if not records:
            return []
        if len(records) * group_size > self.max_slots:
            # Grow once; the next step reuses the larger cache.
            self.max_slots = len(records) * group_size
        total = len(records) * group_size
        if total > self._cache_capacity():
            self._cache = None
        cache = self._ensure_cache(total)
        runner = None
        if self.use_cuda_graph and total <= cache.num_slots - 1:
            if self._graph is None or self._graph.cache is not cache:
                self._graph = GraphDecodeRunner(self.model, cache)
            runner = self._graph
        previous_mhc = os.environ.get("MINIDEEPSEEK_COMPILE_MHC")
        if runner is not None:
            # The Sinkhorn helper is compiled on CUDA; capture the eager loop
            # instead so the graph trace stays deterministic.
            os.environ["MINIDEEPSEEK_COMPILE_MHC"] = "0"
        device = self.model.device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        rng = random.Random(seed)
        thinking_flags = [rng.random() < self.thinking_ratio for _ in records]
        prompts = [
            self.encode_prompt(record, open_thinking=flag)
            for record, flag in zip(records, thinking_flags, strict=True)
        ]
        if any(len(prompt) == 0 for prompt in prompts):
            raise ValueError("every rollout prompt must contain at least one token")
        stop = set(int(t) for t in self.tokenizer.stop_token_ids)

        responses: list[list[int]] = [[] for _ in range(total)]
        logprobs: list[list[float]] = [[] for _ in range(total)]
        finished = [False] * total
        stop_hits = [False] * total
        max_new_hits = [False] * total
        prompt_lengths = [0] * total
        # Per-slot generation budget: never run past the cache context.
        limits = [0] * total

        was_training = self.model.training
        self.model.eval()
        try:
            with _reference_attention():
                for index, prompt in enumerate(prompts):
                    base = index * group_size
                    prompt_lengths[base] = len(prompt)
                    remaining = self.max_context - len(prompt)
                    if remaining <= 0:
                        raise ValueError(
                            f"rollout prompt length {len(prompt)} leaves no room "
                            f"in context {self.max_context}"
                        )
                    limit = min(int(max_new_tokens), remaining)
                    logits = self.model.prefill(prompt, cache, base)
                    last = logits[-1:].expand(group_size, -1)
                    for offset in range(group_size):
                        prompt_lengths[base + offset] = len(prompt)
                        limits[base + offset] = limit
                    for offset in range(1, group_size):
                        cache.copy_slot(base, base + offset)
                    first_ids, first_logprobs = _sample_tokens(
                        last.float(),
                        temperature=float(temperature),
                        top_k=int(top_k),
                        top_p=float(top_p),
                        greedy=False,
                        generator=generator,
                    )
                    for offset in range(group_size):
                        slot = base + offset
                        token = int(first_ids[offset].item())
                        responses[slot].append(token)
                        logprobs[slot].append(float(first_logprobs[offset].item()))
                        if token in stop:
                            finished[slot] = True
                            stop_hits[slot] = True
                        elif limits[slot] <= 1:
                            finished[slot] = True
                            max_new_hits[slot] = True

                active = [slot for slot in range(total) if not finished[slot]]
                while active:
                    slots = torch.tensor(active, dtype=torch.long, device=device)
                    feed = torch.tensor(
                        [responses[slot][-1] for slot in active],
                        dtype=torch.long,
                        device=device,
                    )
                    positions = torch.tensor(
                        [prompt_lengths[slot] + len(responses[slot]) - 1 for slot in active],
                        dtype=torch.long,
                        device=device,
                    )
                    if runner is not None:
                        logits = runner.step(feed, positions, slots)
                    else:
                        logits = self.model.decode_step(feed, positions, cache, slots)
                    next_ids, next_logprobs = _sample_tokens(
                        logits.float(),
                        temperature=float(temperature),
                        top_k=int(top_k),
                        top_p=float(top_p),
                        greedy=False,
                        generator=generator,
                    )
                    still_active: list[int] = []
                    for row, slot in enumerate(active):
                        token = int(next_ids[row].item())
                        responses[slot].append(token)
                        logprobs[slot].append(float(next_logprobs[row].item()))
                        if token in stop:
                            finished[slot] = True
                            stop_hits[slot] = True
                        elif len(responses[slot]) >= limits[slot]:
                            finished[slot] = True
                            max_new_hits[slot] = True
                        else:
                            still_active.append(slot)
                    active = still_active
        finally:
            if previous_mhc is None:
                os.environ.pop("MINIDEEPSEEK_COMPILE_MHC", None)
            else:
                os.environ["MINIDEEPSEEK_COMPILE_MHC"] = previous_mhc
            if was_training:
                self.model.train()

        groups: list[RolloutGroup] = []
        for index, record in enumerate(records):
            group = RolloutGroup(record=record)
            base = index * group_size
            for offset in range(group_size):
                slot = base + offset
                prompt = prompts[index]
                group.samples.append(
                    RolloutSample(
                        prompt_ids=list(prompt),
                        response_ids=list(responses[slot]),
                        sampling_logprobs=list(logprobs[slot]),
                        temperature=float(temperature),
                        record=record,
                        prompt_text=self.tokenizer.decode(prompt),
                        response_text=self.tokenizer.decode(
                            responses[slot], skip_special_tokens=True
                        ),
                        open_thinking=bool(thinking_flags[index]),
                        finished=bool(finished[slot]),
                        max_new_hit=bool(max_new_hits[slot]),
                    )
                )
            groups.append(group)
        return groups

    def _cache_capacity(self) -> int:
        return 0 if self._cache is None else int(self._cache.num_slots)
