from __future__ import annotations

import torch
from torchtitan.config.parallelism import ParallelismConfig

from minideepseek.model.flavors import MODEL_FLAVORS, build_model_config
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
    for flavor in ("mini_60m", "mini_110m", "mini_200m", "mini_60m_dense", "debugmodel"):
        assert flavor in MODEL_FLAVORS
    config = build_model_config(
        "mini_60m", parallelism=ParallelismConfig(enable_sequence_parallel=False), seq_len=512, n_mtp_layers=1
    )
    assert config.vocab_size == 6400
    assert len(config.layers) == 12
    assert config.enable_weight_tying is True
    assert config.compress_ratios[:4] == (0, 0, 4, 32)
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
