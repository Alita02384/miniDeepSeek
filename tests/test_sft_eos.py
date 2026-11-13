"""The SFT terminator up-weighting must be exact at the default and change the loss otherwise."""

from __future__ import annotations

import torch

from minideepseek.model import from_sequences
from minideepseek.model.wrapper import MiniDeepSeekModel


def test_weighted_ce_math():
    losses = torch.tensor([10.0, 1.0, 1.0, 1.0])
    targets = torch.tensor([2, 5, 5, 5])

    plain = MiniDeepSeekModel._weighted_ce(losses, targets, eos_id=None, eos_weight=8.0)
    assert torch.allclose(plain, torch.tensor(3.25))

    weighted = MiniDeepSeekModel._weighted_ce(losses, targets, eos_id=2, eos_weight=4.0)
    # (10*4 + 1 + 1 + 1) / (4 + 1 + 1 + 1)
    assert torch.allclose(weighted, torch.tensor(43.0 / 7.0))

    unit = MiniDeepSeekModel._weighted_ce(losses, targets, eos_id=2, eos_weight=1.0)
    assert torch.allclose(unit, torch.tensor(3.25))


def test_eos_weight_reaches_compute_loss(debug_model):
    batch = from_sequences([[1, 2, 3, 4, 5, 2], [6, 7, 8, 2, 9, 10]])
    base = debug_model.compute_loss(batch, mtp_weight=0.0)
    weighted = debug_model.compute_loss(batch, mtp_weight=0.0, eos_id=2, eos_weight=8.0)
    untouched = debug_model.compute_loss(batch, mtp_weight=0.0, eos_id=None, eos_weight=8.0)

    assert not torch.allclose(base.main, weighted.main)
    assert torch.allclose(base.main, untouched.main)
