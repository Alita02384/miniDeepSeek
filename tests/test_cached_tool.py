"""Cached tool continuation: the append path must match a full forward."""

from __future__ import annotations

import torch

from minideepseek.data.grpo import GRPORecord
from minideepseek.model.batch import from_sequences
from minideepseek.model.wrapper import build_model
from minideepseek.rl.cached_tool import CachedToolRolloutEngine
from minideepseek.tokenizer import load_tokenizer

VOCAB_LIMIT = 2048


def _diff(a, b):
    return float((a.float().detach() - b.float().detach()).abs().max())


def test_feed_appends_tokens_like_full_forward():
    torch.manual_seed(0)
    model = build_model("debugmodel", device="cpu", dtype="float32")
    tokenizer = load_tokenizer(vocab_limit=VOCAB_LIMIT)
    engine = CachedToolRolloutEngine(model, tokenizer, max_slots=2, use_cuda_graph=False)
    prompt = tokenizer.encode(
        tokenizer.render_chat(
            [{"role": "user", "content": "你好"}], add_generation_prompt=True
        )
    )
    cache = engine._ensure_cache(1)
    model.prefill(prompt, cache, 0)
    tokens = [123, 456, 789]
    logits = engine._feed(cache, 0, tokens, start_pos=len(prompt))
    reference = model.forward(from_sequences([prompt + tokens]), use_mtp=False)[0]
    assert _diff(logits[0], reference[-1]) < 1e-4


def test_tool_engine_falls_back_for_plain_records():
    torch.manual_seed(0)
    model = build_model("debugmodel", device="cpu", dtype="float32")
    tokenizer = load_tokenizer(vocab_limit=VOCAB_LIMIT)
    engine = CachedToolRolloutEngine(model, tokenizer, max_slots=4, use_cuda_graph=False)
    record = GRPORecord(prompt=[{"role": "user", "content": "你好"}], domain="general")
    groups = engine.rollout([record], group_size=2, max_new_tokens=2, seed=1)
    assert len(groups) == 1 and len(groups[0].samples) == 2


# ---------------------------------------------------------------------------
# Multi-turn continuation regressions (audit 4.1 / 4.2 / 4.11).
# ---------------------------------------------------------------------------


class _ToolTokenizer:
    """Character tokenizer with an ``open_thinking`` tail.

    The first prompt ends at ``<|assistant|>``; a re-render of the multi-turn
    conversation appends the next generation prompt (and the thinking marker)
    *after* the tool observation, so both thinking flags reproduce the prefix
    exactly -- the same ambiguity as the real tokenizer.
    """

    vocab_size = 2048
    sha256 = "tool-stub"
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    stop_token_ids = [2, 0]

    def __init__(self) -> None:
        self.render_calls: list[bool] = []

    def render_chat(self, messages, *, add_generation_prompt=False, tools=None, open_thinking=False):
        del tools
        self.render_calls.append(bool(open_thinking))
        text = "".join(f"<|{m.get('role')}|>{m.get('content', '')}" for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
            # Only the *next* turn's generation prompt carries the marker; the
            # first prompt must stay a prefix of the re-render.
            if any(m.get("role") == "assistant" for m in messages):
                text += "<think>" if open_thinking else "<think></think>"
        return text

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3 + (ord(char) % 2044) for char in text] or [3]

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr((int(value) - 3) % 2044) for value in ids)


def _tool_record() -> GRPORecord:
    return GRPORecord(
        prompt=[
            {
                "role": "system",
                "content": "",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "calculate_math",
                            "description": "calc",
                            "parameters": {
                                "type": "object",
                                "properties": {"expression": {"type": "string"}},
                                "required": ["expression"],
                            },
                        },
                    }
                ],
            },
            {"role": "user", "content": "1+1"},
        ],
        domain="tool",
        verifier="tool",
        multi_turn=True,
        gt=["2"],
        reference_answer="2",
    )


def _patch_continuation(monkeypatch, *, stop=True):
    import minideepseek.rl.cached_tool as module

    monkeypatch.setattr(
        module,
        "parse_tool_calls",
        lambda text: [{"name": "calculate_math", "arguments": {"expression": "1+1"}}],
    )

    token_id = 2 if stop else 5

    def fake_sample(logits, **kwargs):
        del kwargs
        ids = torch.full(
            (logits.shape[0],), token_id, dtype=torch.long, device=logits.device
        )
        return ids, torch.zeros(logits.shape[0], device=logits.device)

    monkeypatch.setattr(module, "_sample_tokens", fake_sample)


def _expert_counts(model) -> list[int]:
    from minideepseek.train.losses import iter_moe_layers

    return [
        int(moe.router.tokens_per_expert_E.sum().item())
        for _name, moe in iter_moe_layers(model.model)
    ]


def test_continuation_runs_in_eval_and_does_not_touch_expert_usage(monkeypatch):
    from minideepseek.train.losses import reset_expert_usage

    torch.manual_seed(0)
    model = build_model("debugmodel", device="cpu", dtype="float32")
    tokenizer = _ToolTokenizer()
    engine = CachedToolRolloutEngine(
        model,
        tokenizer,
        max_slots=4,
        use_cuda_graph=False,
        max_turns=3,
        thinking_ratio=0.0,
        max_context=64,
    )
    _patch_continuation(monkeypatch)
    reset_expert_usage(model.model)
    model.train()
    engine.rollout([_tool_record()], group_size=1, max_new_tokens=4, seed=1)
    assert model.training is True
    assert _expert_counts(model) == [0] * len(_expert_counts(model))


def test_continuation_preserves_the_recorded_open_thinking(monkeypatch):
    torch.manual_seed(0)
    model = build_model("debugmodel", device="cpu", dtype="float32")
    tokenizer = _ToolTokenizer()
    engine = CachedToolRolloutEngine(
        model,
        tokenizer,
        max_slots=4,
        use_cuda_graph=False,
        max_turns=3,
        thinking_ratio=0.0,  # always closed thinking
        max_context=64,
    )
    _patch_continuation(monkeypatch)
    group = engine.rollout([_tool_record()], group_size=1, max_new_tokens=4, seed=1)[0]
    sample = group.samples[0]
    assert sample.open_thinking is False
    # The last render is the continuation; it must use the sample's flag, not
    # the trial-render order (True first).
    assert tokenizer.render_calls[-1] is False


def test_continuation_stops_before_writing_past_max_context(monkeypatch):
    import minideepseek.rl.cached_tool as module

    torch.manual_seed(0)
    model = build_model("debugmodel", device="cpu", dtype="float32")
    tokenizer = _ToolTokenizer()
    engine = CachedToolRolloutEngine(
        model,
        tokenizer,
        max_slots=4,
        use_cuda_graph=False,
        max_turns=3,
        thinking_ratio=0.0,
        max_context=40,
    )
    _patch_continuation(monkeypatch)
    long_result = {"payload": "x" * 200}
    monkeypatch.setattr(module, "execute_tool", lambda name, args: long_result)
    group = engine.rollout([_tool_record()], group_size=1, max_new_tokens=8, seed=1)[0]
    sample = group.samples[0]
    assert sample.record.rubric["unfinished"] is True
    assert len(sample.prompt_ids) + sample.response_length <= engine.max_context
    assert sample.finished is False


def test_continuation_reaches_max_turns_like_the_reference(monkeypatch):
    torch.manual_seed(0)
    model = build_model("debugmodel", device="cpu", dtype="float32")
    tokenizer = _ToolTokenizer()
    engine = CachedToolRolloutEngine(
        model,
        tokenizer,
        max_slots=4,
        use_cuda_graph=False,
        max_turns=3,
        thinking_ratio=0.0,
        max_context=64,
    )
    _patch_continuation(monkeypatch)
    group = engine.rollout([_tool_record()], group_size=1, max_new_tokens=4, seed=1)[0]
    sample = group.samples[0]
    # turn 0 comes from super().rollout; continuation generates turns 1 and 2.
    assert len(sample.record.rubric["tool_turns"]) == 3
    assert sample.record.rubric["unfinished"] is True


def test_feed_rejects_out_of_context_positions():
    torch.manual_seed(0)
    model = build_model("debugmodel", device="cpu", dtype="float32")
    tokenizer = _ToolTokenizer()
    engine = CachedToolRolloutEngine(
        model, tokenizer, max_slots=2, use_cuda_graph=False, max_context=16
    )
    cache = engine._ensure_cache(1)
    try:
        engine._feed(cache, 0, [1, 2, 3], start_pos=14)
    except RuntimeError as error:
        assert "max_context" in str(error)
    else:  # pragma: no cover - the assertion is the test
        raise AssertionError("out-of-context feed must raise RuntimeError")
