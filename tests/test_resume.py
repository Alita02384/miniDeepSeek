"""Interrupted runs must resume from the newest checkpoint.

This is the "GPU disappeared overnight" contract: the loop finishes the current
step, writes ``final``, and re-running the same command continues from there.
"""

from __future__ import annotations

import json
import os
import signal

import pytest

from minideepseek.config import ModelSettings, TrainConfig, TrainSettings
from minideepseek.model import from_sequences
from minideepseek.train.trainer import (
    _stop_signals,
    build_trainer,
    latest_checkpoint,
)


def _config(tmp_path, **train_overrides) -> TrainConfig:
    train = dict(steps=2, grad_accum=1, log_every=1, out_dir=str(tmp_path / "run"))
    train.update(train_overrides)
    return TrainConfig(
        model=ModelSettings(
            flavor="debugmodel",
            seq_len=64,
            n_mtp_layers=1,
            device="cpu",
            dtype="float32",
            seed=0,
        ),
        train=TrainSettings(**train),
    )


def _batches():
    return [from_sequences([[5, 6, 7, 8], [9, 10]]), from_sequences([[11, 12, 13]])]


def test_latest_checkpoint_prefers_highest_step(tmp_path):
    run = tmp_path / "run"
    for name, step in (("step_000002", 2), ("final", 3), ("step_000004", 4)):
        (run / name).mkdir(parents=True)
        (run / name / "meta.json").write_text(json.dumps({"step": step}), encoding="utf-8")

    assert latest_checkpoint(run) == run / "step_000004"


def test_latest_checkpoint_handles_missing_and_empty(tmp_path):
    assert latest_checkpoint(tmp_path / "nope") is None
    (tmp_path / "empty").mkdir()
    assert latest_checkpoint(tmp_path / "empty") is None


def test_auto_resume_continues_from_checkpoint(tmp_path):
    first, _ = build_trainer(_config(tmp_path), _batches())
    first.train(2)
    assert first.step == 2

    resumed_config = _config(tmp_path, steps=4, auto_resume=True)
    resumed, _ = build_trainer(resumed_config, _batches())
    assert resumed.step == 2
    summary = resumed.train(4)
    assert summary["steps_run"] == 4
    assert summary["interrupted"] is False


def test_auto_resume_off_starts_fresh(tmp_path):
    first, _ = build_trainer(_config(tmp_path), _batches())
    first.train(2)
    fresh, _ = build_trainer(_config(tmp_path), _batches())
    assert fresh.step == 0


def test_explicit_base_outside_out_dir_cannot_overwrite_final(tmp_path):
    first, _ = build_trainer(_config(tmp_path), _batches())
    first.train(2)
    base = tmp_path / "base"
    from minideepseek.model.checkpoint import save_checkpoint

    save_checkpoint(base, first.model, step=0)
    with pytest.raises(ValueError, match="points outside"):
        build_trainer(_config(tmp_path, resume=str(base)), _batches())


def test_stop_finishes_current_step_and_saves(tmp_path):
    trainer, _ = build_trainer(_config(tmp_path, steps=10, ckpt_every=100), _batches())
    original = trainer.train_step

    def stop_after_first():
        record = original()
        trainer.request_stop()
        return record

    trainer.train_step = stop_after_first
    summary = trainer.train(10)

    assert summary["interrupted"] is True
    assert summary["steps_run"] == 1
    # An interrupted run must not look like a finished one: ``final`` is only
    # written when the step target is reached.
    assert not (tmp_path / "run" / "final").exists()
    step_dir = tmp_path / "run" / "step_000001"
    assert step_dir.is_dir()
    assert json.loads((step_dir / "meta.json").read_text())["step"] == 1
    assert latest_checkpoint(tmp_path / "run") == step_dir

    # Re-running continues at step 1 instead of restarting.
    resumed, _ = build_trainer(_config(tmp_path, steps=10, auto_resume=True), _batches())
    assert resumed.step == 1


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_stop_signal_sets_flag_and_restores_handler(tmp_path, signum):
    trainer, _ = build_trainer(_config(tmp_path), _batches())
    before = signal.getsignal(signum)
    with _stop_signals(trainer):
        os.kill(os.getpid(), signum)
    assert trainer._stop_requested is True
    assert signal.getsignal(signum) == before


def test_second_signal_raises_keyboard_interrupt(tmp_path):
    trainer, _ = build_trainer(_config(tmp_path), _batches())
    trainer.request_stop()
    with pytest.raises(KeyboardInterrupt):
        trainer.request_stop()
