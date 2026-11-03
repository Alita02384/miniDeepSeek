from __future__ import annotations

from pathlib import Path

import pytest

from minideepseek.config import EvalConfig, RLConfig, TrainConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative",
    [
        "configs/pretrain/debug_smoke.yaml",
        "configs/pretrain/pilot_0.6b.yaml",
        "configs/pretrain/full_6b.yaml",
        "configs/pretrain/dense_control_1b.yaml",
        "configs/sft/debug_smoke.yaml",
        "configs/sft/math.yaml",
        "configs/sft/code.yaml",
        "configs/sft/world.yaml",
    ],
)
def test_train_configs_load(relative):
    config = load_config(TrainConfig, ROOT / relative)
    assert config.model.flavor
    assert config.train.steps > 0


@pytest.mark.parametrize(
    "relative",
    [
        "configs/grpo/debug_math.yaml",
        "configs/grpo/math.yaml",
        "configs/grpo/code.yaml",
        "configs/grpo/world.yaml",
        "configs/opd/debug_three_teacher.yaml",
        "configs/opd/single_math.yaml",
        "configs/opd/three_teacher.yaml",
    ],
)
def test_rl_configs_load(relative):
    config = load_config(RLConfig, ROOT / relative)
    assert config.rl.method in ("grpo", "opd")


@pytest.mark.parametrize(
    "relative",
    ["configs/eval/debug_math.yaml", "configs/eval/math.yaml", "configs/eval/code.yaml", "configs/eval/world.yaml"],
)
def test_eval_configs_load(relative):
    config = load_config(EvalConfig, ROOT / relative)
    assert config.domain


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  flavor: debugmodel\n  unknown_key: 1\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(TrainConfig, path)
