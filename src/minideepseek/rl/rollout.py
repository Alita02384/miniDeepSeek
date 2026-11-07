"""Torch rollout engine (correctness-first, no KV cache).

The roadmap requires that GRPO/OPD first run on the same model in plain torch
and only later move to vLLM.  This engine recomputes the full prefix at every
decode step, so its log-probabilities are directly comparable with a teacher-
forced actor forward.  It is slow but exact, which is exactly what the
consistency gate needs.

Router decisions can be recorded on the full ``prompt + response`` sequence and
replayed in the actor forward (see ``minideepseek.model.routing``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..data.grpo import GRPORecord
from ..model.batch import PackedBatch, from_sequences
from ..model.routing import RoutingRecord, record_routing
from ..model.wrapper import MiniDeepSeekModel
from ..tokenizer import MiniDeepSeekTokenizer


@dataclass
class RolloutSample:
    prompt_ids: list[int]
    response_ids: list[int]
    sampling_logprobs: list[float]
    temperature: float = 1.0
    record: GRPORecord | None = None
    teacher_key: str | None = None
    prompt_text: str = ""
    response_text: str = ""
    #: Per-response-token loss mask.  Multi-turn agent rollouts mark tool
    #: observations ``False`` so only assistant tokens contribute to loss.
    token_mask: list[bool] | None = None

    @property
    def sequence(self) -> list[int]:
        return self.prompt_ids + self.response_ids

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_ids)

    @property
    def response_length(self) -> int:
        return len(self.response_ids)

    @property
    def num_loss_tokens(self) -> int:
        if self.token_mask is None:
            return len(self.response_ids)
        return sum(1 for value in self.token_mask if value)

    def response_mask(self) -> list[bool]:
        values = self.token_mask if self.token_mask is not None else [True] * len(self.response_ids)
        if len(values) != len(self.response_ids):
            raise ValueError(
                f"token_mask length {len(values)} != response length {len(self.response_ids)}"
            )
        return [False] * len(self.prompt_ids) + list(values)

    def loss_logprobs(self) -> list[float]:
        if self.token_mask is None:
            return list(self.sampling_logprobs)
        return [
            value
            for value, keep in zip(self.sampling_logprobs, self.token_mask, strict=True)
            if keep
        ]


@dataclass
class RolloutGroup:
    record: GRPORecord
    samples: list[RolloutSample] = field(default_factory=list)


class TorchRolloutEngine:
    def __init__(
        self,
        model: MiniDeepSeekModel,
        tokenizer: MiniDeepSeekTokenizer,
        generator=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator

    def encode_prompt(self, record: GRPORecord) -> list[int]:
        text = self.tokenizer.render_chat(
            record.prompt,
            add_generation_prompt=True,
            tools=record.resolved_tools(),
            open_thinking=True,
        )
        return self.tokenizer.encode(text)

    def rollout(
        self,
        records: Sequence[GRPORecord],
        *,
        group_size: int = 4,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> list[RolloutGroup]:
        """Sample ``group_size`` responses for every record."""
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        groups: list[RolloutGroup] = []
        prompt_ids = [self.encode_prompt(record) for record in records]
        repeated = [ids for ids in prompt_ids for _ in range(group_size)]
        generator = self.generator or self.model
        output = generator.generate(
            repeated,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token_ids=self.tokenizer.stop_token_ids,
            seed=seed,
        )
        cursor = 0
        for record, prompt in zip(records, prompt_ids, strict=True):
            group = RolloutGroup(record=record)
            for _ in range(group_size):
                response = output.response_ids[cursor]
                logprobs = output.response_logprobs[cursor]
                group.samples.append(
                    RolloutSample(
                        prompt_ids=list(prompt),
                        response_ids=list(response),
                        sampling_logprobs=list(logprobs),
                        temperature=float(output.temperature),
                        record=record,
                        prompt_text=self.tokenizer.decode(prompt),
                        response_text=self.tokenizer.decode(response, skip_special_tokens=True),
                    )
                )
                cursor += 1
            groups.append(group)
        return groups

    def sync_weights(self, state_dict) -> int:
        """Push actor weights to an external generator; return its version."""
        update = getattr(self.generator, "update_weights", None)
        if update is None:
            return 0
        update(state_dict, getattr(self.generator, "model_path", ""))
        return int(getattr(self.generator, "version", 0))

    @property
    def weight_version(self) -> int:
        return int(getattr(self.generator, "version", 0))

    def check_generator_consistency(
        self,
        samples: Sequence[RolloutSample],
        *,
        temperature: float | None = None,
    ) -> float | None:
        """Compare vLLM sampling log-probs with the actor's teacher-forced ones.

        Returns the max absolute difference, or ``None`` for the pure torch
        backend (where rollout and actor are literally the same forward path).
        """
        if self.generator is None or not samples:
            return None
        actor = self.actor_logprobs(
            samples,
            temperature=temperature or samples[0].temperature,
        ).detach().float().cpu()
        sampling = torch.tensor(
            [value for sample in samples for value in sample.loss_logprobs()],
            dtype=torch.float32,
        )
        if actor.numel() != sampling.numel():
            raise RuntimeError(
                "generator/actor logprob length mismatch: "
                f"{actor.numel()} vs {sampling.numel()}"
            )
        return float((actor - sampling).abs().max())

    # -- actor-side recomputation --------------------------------------------
    @staticmethod
    def build_batch(samples: Sequence[RolloutSample], device: torch.device | str | None = None) -> tuple[PackedBatch, torch.Tensor]:
        sequences = [sample.sequence for sample in samples]
        masks = [sample.response_mask() for sample in samples]
        batch = from_sequences(sequences, target_masks=masks)
        if device is not None:
            batch = batch.to(device)
        response_mask = batch.target_mask.clone()
        return batch, response_mask

    def actor_logprobs(
        self,
        samples: Sequence[RolloutSample],
        *,
        temperature: float = 1.0,
        requires_grad: bool = False,
    ) -> Tensor:
        batch, response_mask = self.build_batch(samples, self.model.device)
        return self.model.response_logprobs(
            batch,
            response_mask,
            requires_grad=requires_grad,
            temperature=temperature,
        )

    @torch.no_grad()
    def record_routing_for(self, samples: Sequence[RolloutSample]) -> RoutingRecord:
        """Record routing on the full rollout sequences (rollout side of replay)."""
        batch, _ = self.build_batch(samples, self.model.device)
        was_training = self.model.training
        self.model.eval()
        try:
            with record_routing(self.model.model) as record:
                self.model.forward(batch, use_mtp=False)
        finally:
            if was_training:
                self.model.train()
        if not record:
            raise RuntimeError("rollout recorded no routers")
        return record

    def flat_logprobs(self, samples: Sequence[RolloutSample]) -> list[float]:
        """Teacher-forced actor log-probs, grouped back per sample."""
        if not samples:
            return []
        values = self.actor_logprobs(samples, temperature=samples[0].temperature)
        values = values.detach().float().tolist()
        result: list[float] = []
        cursor = 0
        for sample in samples:
            result.extend(values[cursor : cursor + sample.response_length])
            cursor += sample.response_length
        if cursor != len(values):
            raise RuntimeError("response-token accounting mismatch")
        return result


def sample_response_indices(groups: Iterable[RolloutGroup]) -> list[int]:
    return [index for group in groups for index in range(len(group.samples))]
