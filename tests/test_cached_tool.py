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
