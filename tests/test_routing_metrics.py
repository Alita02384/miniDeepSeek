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


def test_domain_keys_are_not_cross_rank_reduced():
    """Per-domain diagnostics stay local: ranks can hold different key sets."""
    from minideepseek.train.trainer import _is_reducible

    assert _is_reducible("router_learned_expert_load_max_over_mean", 1.4)
    assert not _is_reducible("router_domain_code_js_vs_pooled", 0.3)
    assert not _is_reducible("router_domain_general_share_max", 0.4)


def test_per_domain_layer_counts_accumulate_over_the_snapshot_window():
    """``tool`` is 3% of the mix, so a single step often contains none.

    The per-domain layer snapshot therefore has to span the whole
    ``train.router_every`` window -- otherwise the dashboard's tool view is
    empty for most snapshots even though the model is routing tool tokens.
    """
    from pathlib import Path

    import torch

    from minideepseek.config import TrainConfig, load_config
    from minideepseek.train.trainer import Trainer

    repo = Path(__file__).resolve().parents[1]
    config = load_config(TrainConfig, repo / "configs" / "pretrain" / "pretrain_145m.yaml")
    assert config.train.router_every > 0
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer._domain_layer_counts = {}
    trainer._domain_batches = {}

    def fake_routing(seed: int):
        generator = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, 6, (64, 2), generator=generator)
        layer = type(
            "Layer",
            (),
            {
                "routing_map": torch.zeros(64, 6, dtype=torch.bool).scatter_(-1, ids, True),
                "is_hash": False,
            },
        )()
        return type("Record", (), {"layers": {"L0": layer}, "num_tokens": 64})()

    # two steps of general + one step that happens to hold no tool tokens
    trainer._accumulate_domain_routing("general", fake_routing(0))
    trainer._accumulate_domain_routing("general", fake_routing(1))
    trainer._accumulate_domain_routing("code", fake_routing(2))
    # the same accumulator feeds the concentration scalars, so a snapshot's
    # per-domain batch counts must match the buckets they came from
    assert trainer._domain_batches == {"general": 2, "code": 1, "overall": 3}
    snap = trainer._domain_layer_snapshot()

    assert sorted(snap) == ["code", "general", "overall"]
    total = lambda name: sum(snap[name]["L0"]["counts"])  # noqa: E731
    # 64 tokens x top-2 slots per call, and a repeated expert id collapses in
    # ``scatter_``, so each call contributes at most 128 routed slots.
    assert 0 < total("general") <= 2 * 128
    assert 0 < total("code") <= 128
    assert total("overall") == total("general") + total("code")
    # the window restarts after a snapshot
    assert trainer._domain_layer_counts == {}
    assert trainer._domain_batches == {}
