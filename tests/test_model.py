from __future__ import annotations

import math

import pytest
import torch

from minideepseek.config import ModelSettings, load_config
from minideepseek.model import build_model, from_sequences
from minideepseek.model.config import (
    MODEL_FLAVORS,
    MoEConfig,
    build_model_config,
    build_model_config_from_settings,
)
from minideepseek.model.model import DeepSeekV4Model
from minideepseek.model.moe import MoE
from minideepseek.train.metrics import iter_limited_swiglu, swiglu_monitor


def test_debug_forward_backward(debug_model, small_batch):
    loss = debug_model.compute_loss(small_batch)
    assert torch.isfinite(loss.total)
    assert loss.main_token_count > 0
    loss.total.backward()
    gradient = sum(
        float(parameter.grad.detach().pow(2).sum())
        for parameter in debug_model.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0


def test_mtp_shapes_and_loss(mtp_debug_model, small_batch):
    outputs = mtp_debug_model.forward(small_batch)
    assert len(outputs) == 2
    assert outputs[0].shape == (small_batch.num_tokens, mtp_debug_model.vocab_size)
    assert outputs[1].shape == outputs[0].shape
    loss = mtp_debug_model.compute_loss(small_batch, mtp_weight=0.3)
    assert set(loss.mtp) == {1}
    assert loss.total >= loss.main  # both terms are positive


def test_parameter_report(debug_model):
    report = debug_model.parameter_report()
    assert report["total"] > 0
    assert report["active_estimate"] > 0
    assert report["moe_routed"] > 0
    assert report["moe_shared"] > 0
    assert report["total"] == sum(p.numel() for p in debug_model.parameters())


def test_token_logprobs_and_logits(debug_model, small_batch):
    logprobs = debug_model.token_logprobs(small_batch)
    assert logprobs.shape == (small_batch.num_tokens,)
    assert logprobs[0].item() == 0.0
    assert torch.all(logprobs[1:] <= 0)
    logits = debug_model.full_logits(small_batch)
    assert logits.shape == (small_batch.num_tokens, debug_model.vocab_size)


def test_generation_is_deterministic_with_greedy(debug_model):
    first = debug_model.generate([[1, 2, 3]], max_new_tokens=5, greedy=True, stop_token_ids=[2])
    second = debug_model.generate([[1, 2, 3]], max_new_tokens=5, greedy=True, stop_token_ids=[2])
    assert first.sequences == second.sequences
    assert len(first.response_ids[0]) == len(first.response_logprobs[0])
    assert all(value <= 0 for value in first.response_logprobs[0])
    # Greedy decoding should match the actor's teacher-forced log-probs.
    from minideepseek.rl.rollout import RolloutSample, TorchRolloutEngine

    class Stub:
        stop_token_ids = [2]

    sample = RolloutSample(
        prompt_ids=[1, 2, 3],
        response_ids=first.response_ids[0],
        sampling_logprobs=first.response_logprobs[0],
    )
    engine = TorchRolloutEngine(debug_model, Stub())
    actor = engine.flat_logprobs([sample])
    assert max(abs(a - b) for a, b in zip(actor, sample.sampling_logprobs)) < 1e-4


def test_mini_flavors_are_registered():
    for flavor in (
        "mini_220m_a80m",
        "mini_60m_dense_d768",
        "mini_145m_a60m",
        "debugmodel",
    ):
        assert flavor in MODEL_FLAVORS
    config = build_model_config("mini_220m_a80m", seq_len=512, n_mtp_layers=1)
    assert config.vocab_size == 6400
    assert len(config.layers) == 7
    assert config.tie_word_embeddings is True
    assert tuple(layer.attention.compress_ratio for layer in config.layers) == (
        0,
        0,
        4,
        32,
        4,
        32,
        4,
    )
    assert config.layers[2].attention.compress_ratio == 4
    assert config.layers[3].attention.compress_ratio == 32
    assert config.n_mtp_layers == 1


def test_swiglu_monitor_reports_clamp_stats(make_debug_model, small_batch):
    model = make_debug_model(swiglu_limit=10.0)
    assert len(list(iter_limited_swiglu(model.model))) > 0
    with swiglu_monitor(model.model) as stats:
        with torch.no_grad():
            model.forward(small_batch)
    assert stats.calls > 0
    assert stats.elements > 0
    assert 0.0 <= stats.as_dict()["swiglu_gate_clamp_rate"] <= 1.0


def test_flavor_parameter_counts_are_calibrated():
    """Lock the three design points: MTP depth 1 + tied embeddings."""
    expected = {
        "debugmodel": 12_984_104,
        "mini_220m_a80m": 216_652_538,
        "mini_60m_dense_d768": 60_876_570,
        "mini_145m_a60m": 145_841_946,
    }
    for flavor, target in expected.items():
        config = build_model_config(
            flavor,
            seq_len=16384 if flavor == "debugmodel" else 4096,
            max_context=16384 if flavor == "debugmodel" else 8192,
            n_mtp_layers=1,
        )
        # meta device: count the parameter tree without allocating ~1GB.
        with torch.device("meta"):
            model = DeepSeekV4Model(config)
        params = sum(p.numel() for p in model.parameters())
        assert abs(params - target) / target < 1e-6, (flavor, params, target)


def test_model_yaml_controls_context_and_mtp():
    settings = load_config(ModelSettings, "configs/model/mini_220m_a80m.yaml")
    assert settings.seq_len == 4096
    assert settings.max_context == 8192
    assert settings.n_mtp_layers == 1
    assert settings.index_topk is None
    config = build_model_config_from_settings(settings)
    assert config.max_context_length == 8192
    assert config.flavor == "mini_220m_a80m"
    # Derived sparse-index budget: 4K -> 1024, 8K -> 2048.
    assert config.layers[2].attention.index_topk == 1024
    import dataclasses

    config_8k = build_model_config_from_settings(
        dataclasses.replace(settings, seq_len=8192, max_context=8192)
    )
    assert config_8k.layers[2].attention.index_topk == 2048


def test_dense_expert_has_unit_routing_weight():
    moe = MoE(
        MoEConfig(
            dim=8,
            hidden_dim=16,
            num_experts=1,
            num_shared_experts=0,
            top_k=1,
            n_hash_layers=0,
            vocab_size=32,
            route_norm=False,
            route_scale=1.0,
            load_balance_coeff=None,
        )
    )
    scores, ids, routing_map = moe.router(torch.randn(4, 8), None)
    assert torch.equal(ids, torch.zeros_like(ids))
    assert torch.allclose(scores, torch.ones_like(scores))
    assert routing_map.shape == (4, 1)


def test_parameter_report_buckets_are_additive(mtp_debug_model):
    report = mtp_debug_model.parameter_report()
    buckets = (
        "embedding",
        "lm_head",
        "attention_and_other",
        "moe_routed",
        "moe_router",
        "moe_shared",
        "mtp",
    )
    assert abs(sum(report[name] for name in buckets) - report["total"]) < 1e-6
    assert report["attention_and_other"] >= 0


def test_forward_rejects_sequence_longer_than_context():
    import pytest

    model = build_model("debugmodel", seq_len=16, n_mtp_layers=0, device="cpu")
    batch = from_sequences([list(range(1, 21))])
    with pytest.raises(ValueError, match="exceeds"):
        model.forward(batch, use_mtp=False)


def test_generate_caps_new_tokens_to_context():
    model = build_model("debugmodel", seq_len=16, n_mtp_layers=0, device="cpu")
    output = model.generate([[1, 2, 3]], max_new_tokens=100, greedy=True)
    assert len(output.response_ids[0]) <= 13


def test_token_logprobs_are_zero_only_at_document_start(debug_model):
    batch = from_sequences([[1, 2, 3]])
    logprobs = debug_model.token_logprobs(batch)
    assert logprobs[0].item() == 0.0
    assert torch.all(logprobs[1:] < 0)


def test_sampling_logprob_uses_unfiltered_policy():
    import torch.nn.functional as F

    from minideepseek.model.wrapper import _sample_tokens

    logits = torch.tensor([[0.0, 10.0, 1.0]])
    ids, logprobs = _sample_tokens(
        logits,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        greedy=False,
        generator=torch.Generator().manual_seed(0),
    )
    assert int(ids.item()) == 1
    expected = F.log_softmax(logits, dim=-1)[0, 1]
    assert torch.allclose(logprobs[0], expected)


def test_three_model_designs_forward_backward():
    """The three architecture files must build and train on the CPU."""
    for flavor in ("mini_60m_dense_d768", "mini_145m_a60m", "mini_220m_a80m"):
        settings = load_config(ModelSettings, f"configs/model/{flavor}.yaml")
        model = build_model(settings, device="cpu", dtype="float32")
        batch = from_sequences([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10]])
        loss = model.compute_loss(batch, mtp_weight=0.3)
        assert torch.isfinite(loss.total)
        loss.total.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
        del model


def test_routed_experts_accumulate_under_autocast():
    """bf16 autocast: expert matmuls return bf16, residual stream stays fp32."""
    moe = MoE(
        MoEConfig(
            dim=8,
            hidden_dim=16,
            num_experts=4,
            num_shared_experts=1,
            top_k=2,
            n_hash_layers=0,
            vocab_size=32,
        )
    )
    x = torch.randn(6, 8)
    ids = torch.randint(0, 32, (6,))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = moe(x, input_ids=ids)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_predictive_entropy_stats_of_known_distributions():
    """Uniform logits give log(V) nats; a peaked distribution gives ~0."""
    from minideepseek.model.batch import from_sequences
    from minideepseek.model.wrapper import MiniDeepSeekModel

    batch = from_sequences([[1, 2, 3, 4, 5, 6]])
    vocab = 64
    tokens = batch.input_ids.numel()
    flat = torch.zeros(tokens, vocab)
    entropy, top1 = MiniDeepSeekModel._masked_distribution_stats(flat, batch, 0)
    assert entropy == pytest.approx(math.log(vocab), rel=1e-4)
    assert top1 == pytest.approx(1.0 / vocab, rel=1e-4)

    peaked = torch.full((tokens, vocab), -1e4)
    peaked[:, 0] = 0.0
    entropy, top1 = MiniDeepSeekModel._masked_distribution_stats(peaked, batch, 0)
    assert entropy < 1e-3
    assert top1 > 0.999
