from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from minideepseek.config import (
    AnticipatoryRoutingSettings,
    ModelSettings,
    OptimizerSettings,
    ScheduleSettings,
    TrainConfig,
    TrainSettings,
)
from minideepseek.data.sft import SFTDataset
from minideepseek.data.sft import iter_batches as sft_batches
from minideepseek.model import from_sequences
from minideepseek.train.distributed import scale_gradients
from minideepseek.train.losses import iter_moe_layers, update_expert_biases
from minideepseek.train.trainer import build_trainer


def _run_metrics(out_dir: Path) -> Path:
    """This launch's metrics file (one run directory per launch)."""
    found = sorted((out_dir / "runs").glob("*/metrics.jsonl"))
    assert len(found) == 1, f"expected exactly one run directory, got {found}"
    return found[0]


def _read_metrics(out_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in _run_metrics(out_dir).read_text().splitlines()
        if line.strip()
    ]


def _config(tmp_path, **train_overrides) -> TrainConfig:
    train = dict(
        steps=2,
        grad_accum=1,
        log_every=1,
        out_dir=str(tmp_path / "run"),
        mtp_weight=0.3,
        balance_loss_coeff=1e-4,
    )
    train.update(train_overrides)
    return TrainConfig(
        model=ModelSettings(
            flavor="debugmodel",
            seq_len=128,
            n_mtp_layers=1,
            swiglu_limit=10.0,
            device="cpu",
            dtype="float32",
            seed=0,
        ),
        optim=OptimizerSettings(lr=1e-3),
        schedule=ScheduleSettings(warmup_steps=1, decay="cosine", min_lr_ratio=0.1),
        train=TrainSettings(**train),
    )


def test_trainer_two_steps_and_checkpoint(tmp_path):
    batches = [from_sequences([[1, 2, 3, 4, 5, 6], [7, 8, 9]]), from_sequences([[10, 11, 12, 13, 14]])]
    config = _config(tmp_path)
    trainer, model = build_trainer(config, batches, tokenizer_hash="stub")
    summary = trainer.train(2)
    assert summary["steps_run"] == 2
    assert summary["last_loss"] is not None
    assert _run_metrics(tmp_path / "run").exists()
    metrics = _read_metrics(tmp_path / "run")
    assert len(metrics) == 2
    assert metrics[-1]["tokens"] > 0
    assert metrics[-1]["swiglu_calls"] > 0
    checkpoint = trainer.save("final")
    assert (checkpoint / "meta.json").exists()
    # The checkpoint must reproduce logits exactly.
    from minideepseek.model import build_model
    from minideepseek.model.checkpoint import load_checkpoint

    reloaded = build_model("debugmodel", seq_len=128, n_mtp_layers=1, device="cpu", swiglu_limit=10.0)
    load_checkpoint(checkpoint, reloaded)
    batch = batches[0]
    with torch.no_grad():
        assert torch.equal(model.forward(batch, use_mtp=False)[0], reloaded.forward(batch, use_mtp=False)[0])


def test_gradient_accumulation(tmp_path):
    batch = from_sequences([[1, 2, 3, 4, 5, 6]])
    config = _config(tmp_path, steps=1, grad_accum=2)
    trainer, _model = build_trainer(config, [batch, batch], tokenizer_hash="stub")
    summary = trainer.train(1)
    assert summary["tokens"] == 2 * int(batch.target_mask.sum())


def test_gradient_accumulation_weights_by_target_tokens(tmp_path):
    """A step's gradient must be the token-weighted mean, not the mean of means.

    The two micro-batches below carry 3 and 5 target tokens, so equal weighting
    would give batch A 5/8 instead of 3/8 of the influence.
    """
    batch_a = from_sequences([[1, 2, 3, 4]], target_masks=[[False, True, True, True]])
    batch_b = from_sequences(
        [[5, 6, 7, 8, 9, 10]], target_masks=[[False, True, True, True, True, True]]
    )
    config = _config(tmp_path, steps=1, grad_accum=2, balance_loss_coeff=0.0)
    trainer, model = build_trainer(
        config, [batch_a, batch_b], tokenizer_hash="stub"
    )
    model.model.zero_grad(set_to_none=True)

    metric_a = trainer._forward_backward(batch_a, 0.3, None, micro_index=0)
    metric_b = trainer._forward_backward(batch_b, 0.3, None, micro_index=1)
    assert (metric_a.tokens, metric_b.tokens) == (3, 5)
    scale = 1.0 / (metric_a.tokens + metric_b.tokens)
    scale_gradients(model.model, scale)

    named = [(name, param) for name, param in model.model.named_parameters() if param.grad is not None]
    actual = {name: param.grad.detach().clone() for name, param in named}

    # Reference: sum of per-token losses over both micro-batches, then / total tokens.
    model.model.zero_grad(set_to_none=True)
    loss_a = trainer.model.compute_loss(batch_a, mtp_weight=0.3).total * metric_a.tokens
    loss_b = trainer.model.compute_loss(batch_b, mtp_weight=0.3).total * metric_b.tokens
    grads = torch.autograd.grad(
        loss_a + loss_b,
        [param for _, param in model.model.named_parameters() if param.requires_grad],
        allow_unused=True,
    )
    expected = {
        name: (grad * scale if grad is not None else None)
        for (name, _), grad in zip(
            [(n, p) for n, p in model.model.named_parameters() if p.requires_grad], grads
        )
    }
    assert actual, "no gradients were accumulated"
    for name, value in actual.items():
        want = expected.get(name)
        if want is None:
            continue
        assert torch.allclose(value, want, atol=1e-6, rtol=1e-4), name


def test_sft_trainer(tmp_path):
    import numpy as np

    from minideepseek.data.io import save_flat_dataset

    dataset_path = tmp_path / "sft.npz"
    input_ids = np.asarray([1, 10, 11, 12, 2], dtype=np.int32)
    save_flat_dataset(
        dataset_path,
        input_ids=input_ids,
        positions=np.arange(len(input_ids), dtype=np.int32),
        target_mask=np.asarray([False, False, True, True, True], dtype=bool),
        offsets=np.asarray([0, len(input_ids)], dtype=np.int64),
        meta={"format": "test"},
    )
    dataset = SFTDataset(dataset_path)
    config = _config(tmp_path, steps=1)
    loader = sft_batches(dataset, batch_tokens=256, shuffle=False)
    trainer, _model = build_trainer(config, loader, tokenizer_hash="stub")
    summary = trainer.train(1)
    assert summary["steps_run"] == 1


def test_expert_bias_update(debug_model, small_batch):
    debug_model.train()
    with torch.no_grad():
        outputs = debug_model.forward(small_batch, use_mtp=False)
        assert outputs
    layers = list(iter_moe_layers(debug_model.model))
    learned = [(name, moe) for name, moe in layers if not moe.router.hash]
    hashed = [(name, moe) for name, moe in layers if moe.router.hash]
    assert learned and hashed
    biases_before = [moe.expert_bias_E.detach().clone() for _name, moe in learned]
    assert any(int(moe.router.tokens_per_expert_E.sum().item()) > 0 for _name, moe in hashed)
    updated = update_expert_biases(debug_model.model)
    assert updated == len(learned)
    for (_name, moe), before in zip(learned, biases_before):
        assert not torch.equal(moe.expert_bias_E, before)
    for _name, moe in layers:
        # Counters are per-step and must be cleared on every layer.
        assert torch.equal(
            moe.router.tokens_per_expert_E,
            torch.zeros_like(moe.router.tokens_per_expert_E),
        )


def test_hash_layers_have_no_expert_bias(debug_model):
    buffer_names = dict(debug_model.model.named_buffers())
    for name, moe in iter_moe_layers(debug_model.model):
        if moe.router.hash:
            assert moe.expert_bias_E is None
            assert f"{name}.expert_bias_E" not in buffer_names
        else:
            assert moe.expert_bias_E is not None
            assert f"{name}.expert_bias_E" in buffer_names


def test_balance_loss_skips_hash_routers(debug_model, small_batch):
    from minideepseek.model.routing import record_routing
    from minideepseek.train.losses import capture_gate_scores, sequence_balance_loss

    debug_model.train()
    debug_model.zero_grad(set_to_none=True)
    with (
        record_routing(debug_model.model) as routing,
        capture_gate_scores(debug_model.model) as gates,
    ):
        debug_model.forward(small_batch, use_mtp=False)
    loss = sequence_balance_loss(
        gates,
        routing,
        num_documents=small_batch.num_documents,
        cu_seqlens=small_batch.cu_seqlens,
        top_k=3,
        num_experts=4,
        coeff=1e-4,
    )
    assert float(loss.detach()) > 0.0
    loss.backward()
    layers = dict(iter_moe_layers(debug_model.model))
    hash_norms = []
    learned_norms = []
    for name in gates:
        moe = layers[name.removesuffix(".router")]
        grad = moe.router.gate.weight.grad
        assert grad is not None
        norm = float(grad.norm())
        (hash_norms if moe.router.hash else learned_norms).append(norm)
    assert hash_norms and learned_norms
    # Hash routers are skipped, so they only see the (tiny) indirect gradient
    # that flows through the learned routers' inputs; the direct balance term
    # used to be the same order as the learned layers.
    assert max(hash_norms) < 0.01 * min(learned_norms)


def test_anticipatory_routing_trainer(tmp_path):
    batch = from_sequences([[1, 2, 3, 4, 5], [6, 7, 8]])
    config = _config(tmp_path, steps=2)
    config.anticipatory = AnticipatoryRoutingSettings(enabled=True, delta=1)
    trainer, model = build_trainer(config, [batch, batch, batch], tokenizer_hash="stub")
    summary = trainer.train(2)
    assert summary["steps_run"] == 2
    assert trainer._ar_state is not None
    assert trainer._ar_state.stale_snapshot() is not None
    # AR prefetches the next step's batch; only batches that a training step
    # consumed may count towards the resume position.
    assert trainer._batches_consumed == 1


def test_factory_loader_cycles_and_validation_is_stable(tmp_path):
    first = from_sequences([[1, 2, 3, 4]])
    second = from_sequences([[5, 6, 7]])
    val = from_sequences([[9, 10, 11]])

    def train_factory():
        yield first
        yield second

    def val_factory():
        yield val

    config = _config(tmp_path, steps=4, val_every=1, val_steps=1)
    trainer, _model = build_trainer(
        config,
        train_factory,
        val_batches=val_factory,
        tokenizer_hash="stub",
    )
    trainer.train(4)
    val_losses = [
        record["val_loss"]
        for record in trainer.logger.history
        if "val_loss" in record
    ]
    assert len(val_losses) == 4
    assert all(torch.isfinite(torch.tensor(value)) for value in val_losses)
    # Validation records must not erase the training metrics from the summary.
    summary = trainer.logger.summary()
    assert summary["steps"] == 4
    assert "loss_mean" in summary
    assert "val_loss_mean" in summary


def test_resume_skips_consumed_batches(tmp_path):
    batches = [from_sequences([[index + 1, index + 2]]) for index in range(6)]
    config = _config(tmp_path, steps=2)
    trainer, _model = build_trainer(config, batches, tokenizer_hash="stub")
    trainer.train(2)
    checkpoint = trainer.save("final")

    resumed_config = _config(tmp_path, steps=4)
    resumed_config.train.resume = str(checkpoint)
    resumed, _model = build_trainer(
        resumed_config, batches, tokenizer_hash="stub"
    )
    resumed.resume(checkpoint)
    assert resumed.step == 2
    assert resumed._batches_consumed == 2
    assert next(resumed._batch_iter).input_ids.tolist() == [3, 4]


def test_resume_replays_data_by_token_offset(tmp_path):
    """The replay offset is measured in tokens, not micro-batches.

    The loader shuffles one global pack order and slices it per rank, so the
    token stream is the same for any GPU count while the micro-batch count is
    not: a 1-GPU run consumes twice as many micro-batches for the same data as
    a 2-GPU run.  A batch counter would skip twice the tokens after moving.
    """
    two = [from_sequences([[index + 1, index + 2]]) for index in range(6)]
    config = _config(tmp_path, steps=1, grad_accum=2)
    trainer, _model = build_trainer(config, two, tokenizer_hash="stub")
    trainer.train(1)
    checkpoint = trainer.save("final")
    assert trainer._data_tokens_global == 4

    # Same 4 tokens per micro-batch as a 2x world size would give.
    four = [
        from_sequences([[index + 1, index + 2, index + 3, index + 4]])
        for index in range(6)
    ]
    resumed, _model = build_trainer(
        _config(tmp_path, steps=4), four, tokenizer_hash="stub"
    )
    resumed.resume(checkpoint)
    assert resumed._data_tokens_global == 4
    assert resumed._batches_consumed == 1
    assert next(resumed._batch_iter).input_ids.tolist() == [2, 3, 4, 5]


def test_resume_legacy_checkpoint_uses_saved_global_batch_tokens(tmp_path):
    """Checkpoints from before ``data_tokens`` fall back to
    ``data_batches * config.data.batch_tokens``, the saved *global* budget."""
    two = [from_sequences([[index + 1, index + 2]]) for index in range(4)]
    config = _config(tmp_path, steps=1, grad_accum=2)
    config.data.batch_tokens = 2
    trainer, _model = build_trainer(config, two, tokenizer_hash="stub")
    trainer.train(1)
    checkpoint = trainer.save("final")

    meta_path = checkpoint / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["extra"].pop("data_tokens") == 4
    meta_path.write_text(json.dumps(meta))

    four = [
        from_sequences([[index + 1, index + 2, index + 3, index + 4]])
        for index in range(4)
    ]
    resumed, _model = build_trainer(
        _config(tmp_path, steps=4), four, tokenizer_hash="stub"
    )
    resumed.resume(checkpoint)
    assert resumed._data_tokens_global == 4
    assert resumed._batches_consumed == 1


def test_resume_rejects_tokenizer_hash_mismatch(tmp_path):
    batches = [from_sequences([[1, 2, 3]])]
    config = _config(tmp_path, steps=1)
    trainer, _model = build_trainer(config, batches, tokenizer_hash="AAA")
    trainer.train(1)
    checkpoint = trainer.save("final")

    other_config = _config(tmp_path, steps=1)
    other, _model = build_trainer(other_config, batches, tokenizer_hash="BBB")
    with pytest.raises(ValueError, match="tokenizer hash mismatch"):
        other.resume(checkpoint)


def test_periodic_checkpoint_step_matches_name(tmp_path):
    batches = [from_sequences([[1, 2, 3]]), from_sequences([[4, 5, 6]])]
    config = _config(tmp_path, steps=2, ckpt_every=1)
    trainer, _model = build_trainer(config, batches, tokenizer_hash="stub")
    trainer.train(2)
    assert json.loads(
        (tmp_path / "run" / "step_000001" / "meta.json").read_text()
    )["step"] == 1


def test_summary_tokens_count_all_steps(tmp_path):
    batches = [from_sequences([[1, 2, 3, 4]])]
    config = _config(tmp_path, steps=4, log_every=2)
    trainer, _model = build_trainer(config, batches, tokenizer_hash="stub")
    summary = trainer.train(4)
    assert summary["tokens"] == 4 * 3  # main-head targets per step


def test_mixed_sources_validate_each_child():
    from minideepseek.train.run import MixedDataset, _validate_dataset

    class Child:
        meta = {"tokenizer_hash": "wrong"}

    tokenizer = type("T", (), {"sha256": "right", "vocab_size": 6400})()
    dataset = MixedDataset(meta={"sources": []}, datasets=[Child()])
    with pytest.raises(ValueError, match="tokenizer hash mismatch"):
        _validate_dataset(dataset, tokenizer, TrainConfig(), label="train")


def test_stage_transition_keeps_fresh_optimizer_and_data(tmp_path):
    batches = [from_sequences([[1, 2, 3]]), from_sequences([[4, 5, 6]])]
    source_config = _config(tmp_path, steps=1)
    source_config.optim.lr = 2e-5
    source, _model = build_trainer(source_config, batches, tokenizer_hash="stub")
    source.train(1)
    checkpoint = source.save("final")

    target_config = _config(tmp_path, steps=2)
    target_config.optim.lr = 2e-6
    target_config.train.resume = str(checkpoint)
    target_config.train.resume_data = False
    target_config.train.resume_optimizer = False
    target_config.train.resume_step = False
    target, _model = build_trainer(target_config, batches, tokenizer_hash="stub")
    target.resume(checkpoint)
    assert target.step == 0
    assert target._batches_consumed == 0
    assert all(
        group["base_lr"] == 2e-6 for group in target.optimizer.param_groups
    )


def test_best_checkpoint_tracks_validation(tmp_path):
    batches = [from_sequences([[1, 2, 3, 4]])]
    val_batches = [from_sequences([[5, 6, 7, 8]])]
    config = _config(tmp_path, steps=2, val_every=1, val_steps=1)
    trainer, _model = build_trainer(
        config, batches, val_batches=val_batches, tokenizer_hash="stub"
    )
    summary = trainer.train(2)
    best = tmp_path / "run" / "best"
    assert (best / "meta.json").exists()
    meta = json.loads((best / "meta.json").read_text())
    assert meta["extra"]["val_loss"] is not None
    assert summary["best_val_loss"] == meta["extra"]["val_loss"]
    assert summary["best_val_step"] is not None


def test_keep_last_ckpt_prunes_old_directories(tmp_path):
    batches = [from_sequences([[1, 2, 3]])]
    config = _config(tmp_path, steps=4, ckpt_every=1, keep_last_ckpt=2)
    trainer, _model = build_trainer(config, batches, tokenizer_hash="stub")
    trainer.train(4)
    retained = sorted(
        path.name
        for path in (tmp_path / "run").iterdir()
        if path.is_dir() and path.name.startswith("step_")
    )
    assert retained == ["step_000003", "step_000004"]


def test_nonfinite_loss_saves_snapshot(tmp_path, monkeypatch):
    from minideepseek.train.trainer import NonFiniteLossError

    class _NanLoss:
        total = torch.tensor(float("nan"), requires_grad=True)
        main = torch.tensor(float("nan"), requires_grad=True)
        mtp = {}
        main_token_count = 1
        mtp_token_count = {}
        indexer = None

    batches = [from_sequences([[1, 2, 3]])]
    config = _config(tmp_path, steps=1, balance_loss_coeff=0.0)
    trainer, _model = build_trainer(config, batches, tokenizer_hash="stub")
    monkeypatch.setattr(
        trainer.model, "compute_loss", lambda batch, mtp_weight=0.3: _NanLoss()
    )
    with pytest.raises(NonFiniteLossError, match="non-finite"):
        trainer.train(1)
    snapshot = tmp_path / "run" / "nan_step_000000"
    assert (snapshot / "meta.json").exists()
    report = json.loads((snapshot / "nan_report.json").read_text())
    assert report["step"] == 0
    assert "metrics" in report


def test_training_report_is_written_during_run(tmp_path):
    batches = [from_sequences([[1, 2, 3, 4]])]
    config = _config(tmp_path, steps=2, report_every=1)
    trainer, _model = build_trainer(config, batches, tokenizer_hash="stub")
    trainer.train(2)
    report = tmp_path / "run" / "report.html"
    assert report.exists()
    document = report.read_text(encoding="utf-8")
    assert "<svg" in document
    assert "miniDeepSeek" in document


def test_router_snapshot_logs_raw_counts_and_bias(tmp_path):
    """``router_every`` must persist the per-layer table + the learned biases."""
    from minideepseek.train.losses import iter_moe_layers

    batches = [from_sequences([[1, 2, 3, 4]])]
    config = _config(tmp_path, steps=2, router_every=1)
    trainer, model = build_trainer(config, batches, tokenizer_hash="stub")
    trainer.train(2)

    records = _read_metrics(tmp_path / "run")
    snapped = [record for record in records if "router_layers" in record]
    assert snapped, "no router snapshot was logged"
    layers = snapped[0]["router_layers"]
    learned = {
        f"{name}.router"
        for name, moe in iter_moe_layers(model.model)
        if getattr(moe, "expert_bias_E", None) is not None
    }
    assert learned, "the debug model should have at least one learned router"
    for name, entry in layers.items():
        assert entry["counts"], f"{name} logged no expert counts"
        assert all(value >= 0 for value in entry["counts"])
        # Only learned routers carry a bias, and it has one value per expert.
        assert ("bias" in entry) is (name in learned)
        if "bias" in entry:
            assert len(entry["bias"]) == len(entry["counts"])

    summary = trainer.logger.summary()
    assert "router_layers_min" not in summary  # structured payloads are skipped
    assert "domain_name_min" not in summary


def test_grad_accum_keeps_router_and_swiglu_metrics(tmp_path):
    """Diagnostics are collected once per step; the merge must not drop them.

    Only the last micro-batch of an accumulation group pays for
    ``routing_metrics``/``swiglu``; intersecting the per-micro-batch key sets
    would then leave nothing and silently erase every MoE panel from the report.
    """
    batches = [from_sequences([[1, 2, 3, 4]]), from_sequences([[5, 6, 7]])]
    config = _config(tmp_path, steps=1, grad_accum=2)
    trainer, _model = build_trainer(config, batches, tokenizer_hash="stub")
    trainer.train(1)

    record = _read_metrics(tmp_path / "run")[-1]
    assert record["tokens"] > 0
    assert "router_learned_expert_load_max_over_mean" in record
    assert "router_learned_min_relative_load" in record
    assert record["swiglu_calls"] > 0


def test_router_snapshot_is_off_by_default(tmp_path):
    batches = [from_sequences([[1, 2, 3, 4]])]
    config = _config(tmp_path, steps=2)
    trainer, _model = build_trainer(config, batches, tokenizer_hash="stub")
    trainer.train(2)
    records = _read_metrics(tmp_path / "run")
    assert not any("router_layers" in record for record in records)


def test_training_report_keeps_records_logged_before_resume(tmp_path):
    """A resumed run must not truncate report.html to the steps it just ran.

    ``auto_resume`` starts with an empty in-memory metric history, so a report
    rendered from memory would drop every step logged before the interruption.
    The earlier run lives in its own directory, and the report stitches it back
    through :func:`minideepseek.runs.lineage`.
    """
    out_dir = tmp_path / "run"
    previous_dir = out_dir / "runs" / "20260101T000000Z_debugmodel_13.0M_1g"
    previous_dir.mkdir(parents=True)
    (previous_dir / "meta.json").write_text(
        json.dumps({"run_id": previous_dir.name, "started_at": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    with (previous_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for step in range(0, 5):
            handle.write(json.dumps({"step": step, "loss": 7.5, "tokens": 10}) + "\n")

    batches = [from_sequences([[1, 2, 3, 4]])]
    config = _config(tmp_path, steps=2, report_every=1)
    trainer, _model = build_trainer(config, batches, tokenizer_hash="stub")
    # Pretend the new launch resumed from the previous run's last step.
    trainer.step = 4
    trainer._write_report()

    document = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "<td>0</td>" in document  # the ancestor's first step survived


def test_trainer_logs_domain_id_from_mixer(tmp_path):
    from minideepseek.data.mix import BatchSource, WeightedBatchMixer

    def factory(tag):
        def generate():
            for _ in range(4):
                yield from_sequences([[tag, 1, 2, 3]])

        return generate

    sources = [
        BatchSource("general", factory(10), weight=0.5),
        BatchSource("code", factory(20), weight=0.5),
    ]
    mixer = WeightedBatchMixer(
        sources, total_batches=2, seed=0, batches_per_step=1, lockstep=True
    )
    config = _config(tmp_path, steps=2, grad_accum=1)
    trainer, _model = build_trainer(config, lambda: iter(mixer), tokenizer_hash="stub")
    trainer.train(2)
    records = _read_metrics(tmp_path / "run")
    assert [record["domain_id"] for record in records] == [0.0, 1.0]
    assert [record["run_id"] for record in records] == [records[0]["run_id"]] * 2


def test_trainer_logs_per_domain_val_loss(tmp_path):
    train_batch = from_sequences([[1, 2, 3, 4, 5, 6]])
    val_general = [from_sequences([[7, 8, 9, 10]])]
    val_code = [from_sequences([[11, 12, 13, 14, 15]])]
    config = _config(tmp_path, steps=1, grad_accum=1, val_every=1, val_steps=1)
    trainer, _model = build_trainer(
        config,
        [train_batch],
        val_batches=lambda: iter(val_general),
        val_sources={
            "general": lambda: iter(val_general),
            "code": lambda: iter(val_code),
        },
        tokenizer_hash="stub",
    )
    trainer.train(1)
    records = _read_metrics(tmp_path / "run")
    val_record = records[-1]
    assert "val_loss" in val_record
    assert "val_loss_general" in val_record
    assert "val_loss_code" in val_record
    assert val_record["val_tokens_general"] > 0
