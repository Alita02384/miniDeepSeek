from __future__ import annotations

import pytest

from minideepseek.config import TrainConfig
from minideepseek.train.curriculum import (
    CurriculumStage,
    _stage_config,
    validate_stages,
)


def _config() -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "model": {"flavor": "debugmodel", "seq_len": 128},
            "data": {"tokenized_path": "base.npz", "batch_tokens": 256},
            "train": {"steps": 10, "out_dir": "checkpoints/base"},
        }
    )


def test_curriculum_stage_validation():
    stages = [
        CurriculumStage(seq_len=2048, tokenized_path="a.npz", until_step=10),
        CurriculumStage(seq_len=4096, tokenized_path="b.npz", until_step=20),
    ]
    validate_stages(stages)
    with pytest.raises(ValueError, match="until_step must increase"):
        validate_stages([stages[0], CurriculumStage(seq_len=4096, tokenized_path="b", until_step=5)])
    with pytest.raises(ValueError, match="must not shrink"):
        validate_stages([stages[1], CurriculumStage(seq_len=2048, tokenized_path="c", until_step=30)])
    with pytest.raises(ValueError, match="seq_len"):
        validate_stages([CurriculumStage(seq_len=99999, tokenized_path="a", until_step=1)])


def test_curriculum_stage_config_overrides(tmp_path):
    base = _config()
    stage = CurriculumStage(
        seq_len=4096,
        tokenized_path="stage.npz",
        until_step=100,
        batch_tokens=1024,
        batch_tokens_final=4096,
        batch_warmup_ratio=0.1,
    )
    config = _stage_config(base, stage, out_dir=tmp_path, index=1, resume="prev/final")
    assert config.model.seq_len == 4096
    assert config.data.tokenized_path == "stage.npz"
    assert config.data.batch_tokens == 1024
    assert config.data.batch_tokens_final == 4096
    assert config.data.batch_warmup_ratio == 0.1
    assert config.train.steps == 100
    assert config.train.resume == "prev/final"
    assert config.train.out_dir.endswith("stage_01_seq4096")
