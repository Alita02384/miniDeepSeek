"""Multi-turn tool/agent rollout.

Refactored from MiniMind ``trainer/train_agent.py::rollout_single`` /
``rollout_batch`` (Apache-2.0).  The key difference from the single-turn
``TorchRolloutEngine`` is that a trajectory is:

    assistant tokens -> <tool_call> -> tool observation -> assistant tokens -> ...

Only assistant-generated tokens are marked for loss; tool observations are
masked out.  The same :class:`RolloutSample` is consumed by GRPO and OPD, so the
teacher/student KL is computed only at assistant positions while the
observation context is still visible.
"""

from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Sequence

from ..data.grpo import GRPORecord
from ..tokenizer import MiniDeepSeekTokenizer
from .rollout import RolloutGroup, RolloutSample, TorchRolloutEngine
from .tools import execute_tool, normalize_tool_args, parse_tool_calls


class ToolRolloutEngine(TorchRolloutEngine):
    """Rollout engine that executes tool calls for multi-turn records.

    Non-tool records (math/code/general RLAIF) fall back to the single-turn
    parent implementation, so one engine can serve a mixed OPD batch.
    """

    def __init__(
        self,
        model,
        tokenizer: MiniDeepSeekTokenizer,
        *,
        generator=None,
        max_turns: int = 3,
        thinking_ratio: float = 0.1,
        max_tool_calls: int = 4,
    ):
        super().__init__(model, tokenizer, generator=generator)
        self.max_turns = max(int(max_turns), 1)
        self.thinking_ratio = float(thinking_ratio)
        self.max_tool_calls = max(int(max_tool_calls), 1)

    def _is_tool_record(self, record: GRPORecord) -> bool:
        return bool(record.resolved_tools()) and (
            record.domain == "tool" or record.multi_turn
        )

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
        groups: list[RolloutGroup] = []
        for index, record in enumerate(records):
            if not self._is_tool_record(record):
                groups.append(
                    self._single_turn(record, group_size, max_new_tokens, temperature, top_k, top_p, seed)
                )
                continue
            group = RolloutGroup(record=record)
            for sample_index in range(group_size):
                group.samples.append(
                    self._rollout_tool(
                        record,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=(None if seed is None else seed + index * group_size + sample_index),
                    )
                )
            groups.append(group)
        return groups

    def _single_turn(self, record, group_size, max_new_tokens, temperature, top_k, top_p, seed):
        base = super().rollout(
            [record],
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )[0]
        return base

    def _rollout_tool(
        self,
        record: GRPORecord,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
    ) -> RolloutSample:
        rng = random.Random(seed)
        messages = [dict(message) for message in record.prompt]
        tools = record.resolved_tools()
        open_thinking = rng.random() < self.thinking_ratio
        prompt_text = self.tokenizer.render_chat(
            messages,
            add_generation_prompt=True,
            tools=tools,
            open_thinking=open_thinking,
        )
        prompt_ids = self.tokenizer.encode(prompt_text)
        response_ids: list[int] = []
        token_mask: list[bool] = []
        sampling_logprobs: list[float] = []
        tool_turns: list[str] = []
        unfinished = False
        final_text = ""
        generator = self.generator or self.model

        for turn in range(self.max_turns):
            context_ids = prompt_ids + response_ids
            output = generator.generate(
                [context_ids],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_token_ids=self.tokenizer.stop_token_ids,
                seed=None if seed is None else seed + turn,
            )
            new_ids = list(output.response_ids[0])
            new_logprobs = list(output.response_logprobs[0])
            new_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            response_ids.extend(new_ids)
            token_mask.extend([True] * len(new_ids))
            sampling_logprobs.extend(new_logprobs)
            tool_turns.append(new_text)
            final_text = new_text
            messages.append({"role": "assistant", "content": new_text})

            calls = parse_tool_calls(new_text)[: self.max_tool_calls]
            if not calls:
                break
            for call in calls:
                name = str(call.get("name", ""))
                args = normalize_tool_args(call)
                result = execute_tool(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            result if result is not None else {"error": "tool not found"},
                            ensure_ascii=False,
                        )[:2048],
                    }
                )
            if turn >= self.max_turns - 1:
                unfinished = True
                break

            # Re-render with the observation.  The chat template is prefix
            # stable here; assert it so a template change cannot silently
            # misalign the training sequence.
            full_text = self.tokenizer.render_chat(
                messages,
                add_generation_prompt=True,
                tools=tools,
                open_thinking=open_thinking,
            )
            full_ids = self.tokenizer.encode(full_text)
            prefix = prompt_ids + response_ids
            if full_ids[: len(prefix)] != prefix:
                raise RuntimeError(
                    "multi-turn chat template is not prefix-stable; observation "
                    "tokens cannot be aligned"
                )
            observation_ids = full_ids[len(prefix) :]
            response_ids.extend(observation_ids)
            token_mask.extend([False] * len(observation_ids))
            sampling_logprobs.extend([0.0] * len(observation_ids))

        record_copy = dataclasses.replace(
            record,
            rubric={
                **record.rubric,
                "tool_turns": tool_turns,
                "unfinished": unfinished,
            },
        )
        return RolloutSample(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            sampling_logprobs=sampling_logprobs,
            temperature=float(temperature),
            record=record_copy,
            prompt_text=prompt_text,
            response_text=final_text,
            token_mask=token_mask,
        )
