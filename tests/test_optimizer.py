from __future__ import annotations

import math

import torch

from minideepseek.config import OptimizerSettings
from minideepseek.train.optimizer import (
    HybridOptimizer,
    Muon,
    build_optimizer,
    zeropower_via_newtonschulz5,
)
from minideepseek.train.scheduler import WarmupDecaySchedule


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    matrix = torch.randn(32, 16)
    orthogonal = zeropower_via_newtonschulz5(matrix, steps=5)
    assert orthogonal.shape == matrix.shape
    singular_values = torch.linalg.svdvals(orthogonal.float())
    original = torch.linalg.svdvals(matrix.float())
    # NS is an approximate orthogonalizer at 5 steps: singular values are
    # pulled into a narrow band around 1, far tighter than the input's.
    assert float(singular_values.max()) < 1.5
    assert float(singular_values.min()) > 0.3
    assert float(singular_values.std()) < float(original.std())


def test_muon_step_changes_parameter():
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = Muon([parameter], lr=0.05, momentum=0.9)
    loss = (parameter**2).sum()
    loss.backward()
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(before, parameter.detach())
    assert torch.isfinite(parameter).all()


def test_hybrid_optimizer_covers_all_parameters(debug_model):
    optimizer = build_optimizer(debug_model, OptimizerSettings(lr=1e-3))
    assert isinstance(optimizer, HybridOptimizer)
    grouped = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in debug_model.parameters() if parameter.requires_grad}
    assert grouped == expected
    assert optimizer.muon is not None and optimizer.adamw is not None


def test_scheduler_warmup_and_decay():
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = Muon([parameter], lr=1.0)
    optimizer.param_groups[0]["base_lr"] = 1.0
    schedule = WarmupDecaySchedule(optimizer, warmup_steps=2, total_steps=10, decay="cosine", min_lr_ratio=0.1)
    assert math.isclose(schedule.factor(0), 0.5)
    assert math.isclose(schedule.factor(1), 1.0)
    assert math.isclose(schedule.factor(9), 0.1, rel_tol=1e-6)
    schedule.step(0)
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.5)
