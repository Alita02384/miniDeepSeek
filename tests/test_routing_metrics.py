"""Per-domain expert statistics: what the whole-batch average cannot show."""

from __future__ import annotations

import pytest
import torch

from minideepseek.model.routing import domain_expert_specialization


def test_domain_expert_specialization_separates_domains():
    """A domain that pulls its own experts must show up as JS > 0."""
    layer = "layers.2.feed_forward.router"
    general = torch.tensor([300.0, 300.0, 300.0, 300.0, 300.0, 300.0])
    code = torch.tensor([50.0, 50.0, 50.0, 700.0, 50.0, 50.0])
    out = domain_expert_specialization({"general": {layer: general}, "code": {layer: code}})
    assert out["router_domain_general_share_max"] < out["router_domain_code_share_max"]
    assert out["router_domain_code_share_max"] > 1 / 6
    assert out["router_domain_general_js_vs_pooled"] > 0
    assert out["router_domain_code_js_vs_pooled"] > out["router_domain_general_js_vs_pooled"]

    # Identical distributions cannot diverge from their own pool.
    flat = domain_expert_specialization(
        {"a": {layer: general}, "b": {layer: general.clone()}}
    )
    assert flat["router_domain_a_js_vs_pooled"] == pytest.approx(0.0, abs=1e-6)


def test_group_metrics_entropy_is_bounded():
    """Normalised routing entropy is in [0, 1] whatever the load looks like."""
    from minideepseek.model.routing import LayerRouting, _group_metrics

    counts = [1462, 6724, 5297, 116, 2760, 25]  # the skewed step-1650 snapshot
    rows = [
        torch.nn.functional.one_hot(torch.tensor(expert), 6).float().repeat(n, 1)
        for expert, n in enumerate(counts)
    ]
    layer = LayerRouting(
        name="l", layer_id=0, is_hash=False, expert_ids=None, scores=None,
        routing_map=torch.cat(rows),
    )
    values = [float(v) for v in _group_metrics([layer])]
    normalized = values[3]
    nats = values[4]
    assert 0.0 <= normalized <= 1.0
    assert nats == pytest.approx(1.2912, abs=1e-3)
