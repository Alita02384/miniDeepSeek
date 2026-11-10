"""KV-cache decode must reproduce the full-recompute forward exactly.

These tests are the gate for the rollout engine: if the incremental path
drifts, every GRPO ratio and OPD KL computed on its rollouts is wrong.
"""

from __future__ import annotations

import pytest
import torch

from minideepseek.data.grpo import GRPORecord
from minideepseek.model.batch import from_sequences
from minideepseek.model.cache import RolloutCache
from minideepseek.model.wrapper import build_model
from minideepseek.rl.cached_rollout import CachedRolloutEngine
from minideepseek.tokenizer import load_tokenizer

VOCAB_LIMIT = 2048


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    instance = build_model("debugmodel", device="cpu", dtype="float32")
    instance.eval()
    return instance


def _full_logits(model, sequence):
    batch = from_sequences([list(sequence)])
    return model.forward(batch, use_mtp=False)[0]


def _diff(a, b):
    return float((a.float() - b.float()).abs().max())


def test_prefill_and_decode_match_full_forward(model):
    prompt = list(range(100, 300))
    cache = RolloutCache.from_model(model, num_slots=2)
    logits = model.prefill(prompt, cache, 0)
    reference = _full_logits(model, prompt)
    assert _diff(logits[-1], reference[-1]) < 1e-5
    # CSA and HCA layers both wrote compressed entries.
    ratios = {layer.compress_ratio: int(layer.n_comp[0]) for layer in cache.layers.values()}
    assert ratios[4] == len(prompt) // 4
    assert ratios[128] == len(prompt) // 128

    sequence = list(prompt)
    for step in range(3):
        token = 900 + step
        sequence.append(token)
        decoded = model.decode_step(
            torch.tensor([token]),
            torch.tensor([len(sequence) - 1]),
            cache,
            torch.tensor([0]),
        )
        expected = _full_logits(model, sequence)
        assert _diff(decoded[0], expected[-1]) < 1e-5


def test_hca_multiple_compressed_entries(model):
    prompt = list(range(100, 700))
    cache = RolloutCache.from_model(model, num_slots=1)
    logits = model.prefill(prompt, cache, 0)
    reference = _full_logits(model, prompt)
    assert _diff(logits[-1], reference[-1]) < 1e-5
    hca = [layer for layer in cache.layers.values() if layer.compress_ratio == 128]
    assert hca and int(hca[0].n_comp[0]) == len(prompt) // 128


def test_batched_slots_and_fork(model):
    cache = RolloutCache.from_model(model, num_slots=4)
    prompts = [list(range(10, 210)), list(range(300, 360))]
    for slot, prompt in enumerate(prompts):
        cache.reset(torch.tensor([slot]))
        logits = model.prefill(prompt, cache, slot)
        assert _diff(logits[-1], _full_logits(model, prompt)[-1]) < 1e-5
    cache.copy_slot(0, 2)
    sequences = [list(prompts[0]), list(prompts[1]), list(prompts[0])]
    slots = torch.tensor([0, 1, 2])
    for step in range(2):
        tokens = [700 + step, 60 + step, 800 + step]
        positions = [len(sequence) for sequence in sequences]
        for index, token in enumerate(tokens):
            sequences[index].append(token)
        decoded = model.decode_step(
            torch.tensor(tokens), torch.tensor(positions), cache, slots
        )
        for index, sequence in enumerate(sequences):
            assert _diff(decoded[index], _full_logits(model, sequence)[-1]) < 1e-5


def test_cached_engine_logprobs_match_actor():
    torch.manual_seed(0)
    model = build_model("debugmodel", device="cpu", dtype="float32")
    tokenizer = load_tokenizer(vocab_limit=VOCAB_LIMIT)
    engine = CachedRolloutEngine(model, tokenizer, max_slots=8)
    records = [
        GRPORecord(prompt=[{"role": "user", "content": "1+1等于几？"}], domain="math"),
        GRPORecord(prompt=[{"role": "user", "content": "你好"}], domain="general"),
    ]
    groups = engine.rollout(records, group_size=2, max_new_tokens=5, seed=7)
    samples = [sample for group in groups for sample in group.samples]
    assert len(samples) == 4
    for sample in samples:
        assert len(sample.sampling_logprobs) == len(sample.response_ids)
        assert all(value == value for value in sample.sampling_logprobs)
    actor = engine.actor_logprobs(samples)
    sampling = torch.tensor(
        [value for sample in samples for value in sample.sampling_logprobs]
    )
    assert _diff(actor, sampling) < 1e-4


def test_cached_engine_respects_max_new_tokens_and_stop():
    torch.manual_seed(0)
    model = build_model("debugmodel", device="cpu", dtype="float32")
    tokenizer = load_tokenizer(vocab_limit=VOCAB_LIMIT)
    engine = CachedRolloutEngine(model, tokenizer, max_slots=4)
    record = GRPORecord(prompt=[{"role": "user", "content": "你好"}], domain="general")
    groups = engine.rollout([record], group_size=3, max_new_tokens=2, seed=1)
    for sample in groups[0].samples:
        assert 1 <= len(sample.response_ids) <= 2
