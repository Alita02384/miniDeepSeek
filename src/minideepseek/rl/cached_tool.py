# Copyright (c) 2026 miniDeepSeek contributors.
"""Multi-turn tool rollout on top of the KV-cache engine.

``ToolRolloutEngine`` re-prefills the whole trajectory after every tool
observation, which makes tool GRPO as slow as the old torch rollout.  This
class keeps the per-turn cache: the first turn is batched for all group
samples, then each sample's tool observation is appended token by token
(``decode_step``) and generation continues from the same cache slot.

The chat template is prefix-stable (asserted, as in ``ToolRolloutEngine``), so
the appended observation tokens are exactly the tokens the next turn would see
in a full re-render.  Observation tokens are masked out of the loss and get
zero sampling log-prob, matching the SFT/agent convention.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence

import torch

from ..data.grpo import GRPORecord
from ..model.wrapper import _sample_tokens
from ..tokenizer import MiniDeepSeekTokenizer
from .cached_rollout import CachedRolloutEngine
from .rollout import RolloutGroup, RolloutSample
from .tools import execute_tool, normalize_tool_args, parse_tool_calls


class CachedToolRolloutEngine(CachedRolloutEngine):
    """Cached single-turn path + per-slot tool continuation."""

    def __init__(
        self,
        model,
        tokenizer: MiniDeepSeekTokenizer,
        *,
        max_turns: int = 3,
        thinking_ratio: float = 0.1,
        max_tool_calls: int = 4,
        **kwargs,
    ):
        super().__init__(model, tokenizer, thinking_ratio=thinking_ratio, **kwargs)
        self.max_turns = max(int(max_turns), 1)
        self.max_tool_calls = max(int(max_tool_calls), 1)
        self.thinking_ratio = float(thinking_ratio)

    def _is_tool_record(self, record: GRPORecord) -> bool:
        return bool(record.resolved_tools()) and (
            record.domain == "tool" or record.multi_turn
        )

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
        """Batch plain records together; batch every tool record's first turn.

        Calling ``super().rollout`` once per record would reset the cache and
        recapture the CUDA graph per prompt (a mixed OPD batch spent ~40 s in
        prefill alone).  The first turn of all tool records can share one
        batched call; only the tool-observation continuations are per slot.
        """
        results: list[RolloutGroup | None] = [None] * len(records)
        plain = [index for index, record in enumerate(records) if not self._is_tool_record(record)]
        tool = [index for index, record in enumerate(records) if self._is_tool_record(record)]
        if plain:
            groups = super().rollout(
                [records[index] for index in plain],
                group_size=group_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
            )
            for index, group in zip(plain, groups, strict=True):
                results[index] = group
        if tool:
            tool_records = [records[index] for index in tool]
            base_groups = super().rollout(
                tool_records,
                group_size=group_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=None if seed is None else seed + 1,
            )
            cache = self._cache
            if cache is None:
                raise RuntimeError("tool continuation needs the rollout cache")
            # ``super().rollout`` restores the model to train() on exit.  The
            # continuation decode must run under eval(): MoE routers only
            # accumulate ``tokens_per_expert_E`` in training mode, and a CUDA
            # graph first captured here would bake that accumulation into the
            # replay.  Keep the eval window around every continuation call.
            was_training = self.model.training
            self.model.eval()
            try:
                for local, (index, base_group) in enumerate(
                    zip(tool, base_groups, strict=True)
                ):
                    group = RolloutGroup(record=records[index])
                    for offset, sample in enumerate(base_group.samples):
                        slot = local * group_size + offset
                        group.samples.append(
                            self._continue_tool(
                                records[index],
                                sample,
                                slot,
                                cache,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                seed=None if seed is None else seed + index * group_size + offset,
                            )
                        )
                    results[index] = group
            finally:
                if was_training:
                    self.model.train()
        return [group for group in results if group is not None]

    def _feed(
        self,
        cache,
        slot: int,
        tokens: Sequence[int],
        *,
        start_pos: int,
    ):
        device = self.model.device
        if start_pos < 0 or start_pos + len(tokens) > self.max_context:
            raise RuntimeError(
                f"tool continuation would write positions {start_pos}.."
                f"{start_pos + len(tokens) - 1} outside max_context="
                f"{self.max_context}"
            )
        runner = self._graph
        logits = None
        for offset, token in enumerate(tokens):
            ids = torch.tensor([int(token)], dtype=torch.long, device=device)
            positions = torch.tensor([start_pos + offset], dtype=torch.long, device=device)
            slots = torch.tensor([slot], dtype=torch.long, device=device)
            if runner is not None:
                logits = runner.step(ids, positions, slots)
            else:
                logits = self.model.decode_step(ids, positions, cache, slots)
        return logits

    def _execute_calls(self, messages: list[dict], calls: list[dict]) -> None:
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

    def _continue_tool(
        self,
        record: GRPORecord,
        sample: RolloutSample,
        slot: int,
        cache,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
    ) -> RolloutSample:
        if self.model.training:
            raise RuntimeError("tool continuation must run under model.eval()")
        responses = list(sample.response_ids)
        logprobs = list(sample.sampling_logprobs)
        token_mask = [True] * len(responses)
        messages = [dict(message) for message in record.prompt]
        messages.append({"role": "assistant", "content": sample.response_text})
        tool_turns: list[str] = [sample.response_text]
        unfinished = bool(sample.max_new_hit)
        last_stopped = bool(sample.finished) and not unfinished
        final_text = sample.response_text
        stop = set(int(t) for t in self.tokenizer.stop_token_ids)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.model.device)
            generator.manual_seed(int(seed))
        open_thinking = bool(sample.open_thinking)
        record_mismatch = False

        for turn in range(1, self.max_turns):
            calls = parse_tool_calls(final_text)[: self.max_tool_calls]
            if not calls:
                break
            self._execute_calls(messages, calls)
            prefix = sample.prompt_ids + responses
            # Re-render with the flag the prompt was actually built with.  The
            # two scaffolds differ only in the generation tail, so matching the
            # prefix cannot distinguish them.
            full_text = self.tokenizer.render_chat(
                messages,
                add_generation_prompt=True,
                tools=record.resolved_tools(),
                open_thinking=open_thinking,
            )
            full_ids = self.tokenizer.encode(full_text)
            if full_ids[: len(prefix)] != prefix:
                # The ChatML template normalizes assistant content (it always
                # re-renders the closed think scaffold, strips leading
                # newlines, and U+FFFD pieces cannot round-trip).  A legitimate
                # open-thinking turn therefore need not be prefix-stable.  Do
                # not kill the whole GRPO run: keep the sampled trajectory and
                # mark it unfinished instead of feeding a misaligned suffix.
                unfinished = True
                last_stopped = False
                record_mismatch = True
                break
            observation_ids = full_ids[len(prefix) :]
            cache_len = int(cache.length[slot].item())
            if cache_len > len(prefix):
                raise RuntimeError(
                    f"cache length {cache_len} exceeds rendered prefix {len(prefix)}"
                )
            # Leave at least one context slot for the next assistant token.  If
            # the observation cannot fit, stop the trajectory instead of
            # letting decode write past max_context (RoPE would raise
            # IndexError, and a truncated observation is not a valid turn).
            room = self.max_context - len(prefix) - len(observation_ids)
            if room <= 0:
                unfinished = True
                last_stopped = False
                break
            # The stop token that ended the previous turn was sampled but not
            # fed back; replay the missing prefix tail, then the observation.
            tokens_to_feed = prefix[cache_len:] + observation_ids
            logits = self._feed(
                cache,
                slot,
                tokens_to_feed,
                start_pos=cache_len,
            )
            responses.extend(observation_ids)
            logprobs.extend([0.0] * len(observation_ids))
            token_mask.extend([False] * len(observation_ids))

            turn_limit = min(int(max_new_tokens), room)
            new_tokens: list[int] = []
            new_logprobs: list[float] = []
            stopped = False
            for step in range(turn_limit):
                next_ids, next_logprobs = _sample_tokens(
                    logits.float(),
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    greedy=False,
                    generator=generator,
                )
                token = int(next_ids[0].item())
                new_tokens.append(token)
                new_logprobs.append(float(next_logprobs[0].item()))
                if token in stop:
                    stopped = True
                    break
                if step + 1 < turn_limit:
                    new_len = int(cache.length[slot].item())
                    logits = self._feed(cache, slot, [token], start_pos=new_len)
            responses.extend(new_tokens)
            logprobs.extend(new_logprobs)
            token_mask.extend([True] * len(new_tokens))
            final_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            tool_turns.append(final_text)
            messages.append({"role": "assistant", "content": final_text})
            last_stopped = stopped
            if not stopped:
                # The assistant turn was cut off mid-generation.  Do not parse
                # a partial tool call as if it were complete.
                unfinished = True
                break
            if turn >= self.max_turns - 1:
                # Last assistant turn: execute its calls but do not feed another
                # observation.  The reference engine does the same, so both
                # paths produce ``max_turns`` assistant turns.
                last_calls = parse_tool_calls(final_text)[: self.max_tool_calls]
                if last_calls:
                    self._execute_calls(messages, last_calls)
                    unfinished = True
                break

        rubric = {**record.rubric, "tool_turns": tool_turns, "unfinished": unfinished}
        if record_mismatch:
            rubric["continuation_render_mismatch"] = True
        record_copy = dataclasses.replace(record, rubric=rubric)
        return RolloutSample(
            prompt_ids=list(sample.prompt_ids),
            response_ids=responses,
            sampling_logprobs=logprobs,
            temperature=float(temperature),
            record=record_copy,
            prompt_text=sample.prompt_text,
            response_text=final_text,
            token_mask=token_mask,
            open_thinking=open_thinking,
            finished=bool(last_stopped and not unfinished),
            max_new_hit=bool(not last_stopped or unfinished),
        )
