"""mHC Sinkhorn: eager/compiled parity and doubly-stochastic structure."""

from __future__ import annotations

import torch

from minideepseek.model.mhc import HcSplitSinkhorn, _run_sinkhorn, _sinkhorn_normalise


def _reference(comb, iters, eps):
    for _ in range(iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def test_sinkhorn_matches_reference():
    torch.manual_seed(0)
    comb = torch.rand(64, 4, 4) + 0.1
    got = _sinkhorn_normalise(comb.clone(), 20, 1e-6)
    expected = _reference(comb.clone(), 20, 1e-6)
    torch.testing.assert_close(got, expected)


def test_dispatch_uses_eager_on_cpu():
    torch.manual_seed(0)
    comb = torch.rand(32, 4, 4) + 0.1
    # On CPU the compiled path is bypassed, so this must equal eager exactly.
    torch.testing.assert_close(
        _run_sinkhorn(comb.clone(), 20, 1e-6), _sinkhorn_normalise(comb.clone(), 20, 1e-6)
    )


def test_hc_split_sinkhorn_is_doubly_stochastic():
    torch.manual_seed(0)
    module = HcSplitSinkhorn(hc_mult=4, sinkhorn_iters=20, eps=1e-6)
    mixes = torch.randn(128, (2 + 4) * 4)
    scale = torch.ones(3)
    base = torch.zeros((2 + 4) * 4)
    pre, post, comb = module(mixes, scale, base)
    assert pre.shape == (128, 4) and post.shape == (128, 4)
    assert comb.shape == (128, 4, 4)
    # The loop ends on a column normalisation, so columns are exact and rows
    # are only approximately 1.
    torch.testing.assert_close(comb.sum(-1), torch.ones(128, 4), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(comb.sum(-2), torch.ones(128, 4), atol=1e-5, rtol=1e-5)
