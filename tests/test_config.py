from __future__ import annotations

from pathlib import Path

import pytest

from minideepseek.config import EvalConfig, ModelSettings, RLConfig, TrainConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative",
    [
        "configs/pretrain/debug_smoke.yaml",
        "configs/pretrain/pilot_0.6b.yaml",
        "configs/pretrain/full_6b.yaml",
        "configs/pretrain/dense_control_1b.yaml",
        "configs/sft/debug_smoke.yaml",
        "configs/sft/code.yaml",
        "configs/sft/general.yaml",
        "configs/sft/tool.yaml",
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
        "configs/grpo/code.yaml",
        "configs/grpo/general.yaml",
        "configs/grpo/tool.yaml",
        "configs/opd/debug_three_teacher.yaml",
        "configs/opd/three_teacher.yaml",
        "configs/opd/two_teacher_general_tool.yaml",
    ],
)
def test_rl_configs_load(relative):
    config = load_config(RLConfig, ROOT / relative)
    assert config.rl.method in ("grpo", "opd")


@pytest.mark.parametrize(
    "relative",
    [
        "configs/eval/debug_math.yaml",
        "configs/eval/code.yaml",
        "configs/eval/general.yaml",
        "configs/eval/tool.yaml",
    ],
)
def test_eval_configs_load(relative):
    config = load_config(EvalConfig, ROOT / relative)
    assert config.domain


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  flavor: debugmodel\n  unknown_key: 1\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(TrainConfig, path)


@pytest.mark.parametrize(
    "name, dim, layers, experts",
    [
        ("mini_220m_a80m", 768, 7, 8),
        ("mini_60m_dense_d768", 768, 7, 1),
        ("mini_145m_a60m", 768, 7, 6),
        ("debugmodel", 256, 4, 4),
    ],
)
def test_model_configs_are_complete(name, dim, layers, experts):
    settings = load_config(ModelSettings, ROOT / f"configs/model/{name}.yaml")
    assert settings.flavor == name
    assert settings.dim == dim
    assert settings.n_layers == layers
    assert settings.num_experts == experts
    assert len(settings.compress_ratios) == layers


def test_train_config_merges_model_file(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "model_file: configs/model/mini_145m_a60m.yaml\n"
        "model:\n"
        "  seq_len: 2048\n"
        "  n_mtp_layers: 0\n"
    )
    config = load_config(TrainConfig, path)
    assert config.model.flavor == "mini_145m_a60m"
    assert config.model.dim == 768
    assert config.model.num_experts == 6
    assert config.model.seq_len == 2048
    assert config.model.n_mtp_layers == 0
    assert config.model.max_context == 8192


def test_model_file_override_changes_architecture():
    config = load_config(
        TrainConfig,
        ROOT / "configs/pretrain/full_6b.yaml",
        model_file=ROOT / "configs/model/mini_60m_dense_d768.yaml",
    )
    assert config.model.flavor == "mini_60m_dense_d768"
    assert config.model.num_experts == 1
    assert config.model.top_k == 1
    assert config.model.seq_len == 4096
