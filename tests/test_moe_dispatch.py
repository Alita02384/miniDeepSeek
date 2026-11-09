"""Equivalence of the sorted expert dispatch against an explicit per-token loop."""

from __future__ import annotations

import pytest
import torch

from minideepseek.model.config import MoEConfig
from minideepseek.model.moe import RoutedExperts


def _reference(x, scores, ids, w13, w2, activation):
    out = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate_up = torch.einsum("d,efd->ef", x[token], w13[expert])
            hidden = activation(gate_up[0], gate_up[1])
            out[token] += torch.einsum("f,df->d", hidden, w2[expert]) * scores[token, slot]
    return out


def _module(dtype):
    config = MoEConfig(
        dim=16,
        hidden_dim=24,
        num_experts=4,
        num_shared_experts=0,
        top_k=2,
        n_hash_layers=0,
        layer_id=0,
        vocab_size=64,
    )
    module = RoutedExperts(config).to(dtype)
    with torch.no_grad():
        module.w13.weight.normal_(0, 0.1)
        module.w2.weight.normal_(0, 0.1)
    tokens = 11
    x = torch.randn(tokens, config.dim, dtype=dtype, requires_grad=True)
    scores = torch.rand(tokens, config.top_k, dtype=torch.float32, requires_grad=True)
    ids = torch.stack(
        [torch.randint(0, config.num_experts, (tokens,)) for _ in range(config.top_k)], dim=-1
    ).to(torch.int32)
    return module, config, x, scores, ids


def test_sorted_dispatch_matches_explicit_loop():
    torch.manual_seed(0)
    module, _config, x, scores, ids = _module(torch.float32)
    got = module(x, scores, ids)
    expected = _reference(
        x, scores, ids, module.w13.weight, module.w2.weight, module.activation_fn
    )
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    # Backward must reach tokens, routing weights and every used expert weight.
    got.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    for expert in ids.reshape(-1).unique().tolist():
        grad = module.w13.weight.grad[expert]
        assert grad is not None and grad.abs().sum() > 0


def test_sorted_dispatch_bf16_matches_fp32_reference():
    torch.manual_seed(0)
    module, _config, x, scores, ids = _module(torch.bfloat16)
    got = module(x, scores, ids)
    expected = _reference(
        x.float(), scores, ids, module.w13.weight.float(), module.w2.weight.float(),
        module.activation_fn,
    )
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=5e-2)


def test_sorted_dispatch_handles_empty_expert():
    config = MoEConfig(
        dim=8,
        hidden_dim=16,
        num_experts=4,
        num_shared_experts=0,
        top_k=1,
        n_hash_layers=0,
        layer_id=0,
        vocab_size=32,
    )
    module = RoutedExperts(config)
    x = torch.randn(3, config.dim)
    scores = torch.ones(3, 1)
    ids = torch.tensor([[0], [0], [0]], dtype=torch.int32)
    out = module(x, scores, ids)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
