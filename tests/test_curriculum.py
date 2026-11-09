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
        validate_stages(
            [stages[0], CurriculumStage(seq_len=4096, tokenized_path="b", until_step=5)]
        )
    with pytest.raises(ValueError, match="must not shrink"):
        validate_stages(
            [stages[1], CurriculumStage(seq_len=2048, tokenized_path="c", until_step=30)]
        )
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
    config = _stage_config(
        base, stage, out_dir=tmp_path, index=1, resume="prev/final", schedule_steps=1000
    )
    assert config.model.seq_len == 4096
    assert config.data.tokenized_path == "stage.npz"
    assert config.data.batch_tokens == 1024
    assert config.data.batch_tokens_final == 4096
    assert config.data.batch_warmup_ratio == 0.1
    assert config.train.steps == 100
    assert config.train.schedule_steps == 1000
    assert config.train.resume == "prev/final"
    assert config.train.out_dir.endswith("stage_01_seq4096")


def test_curriculum_stage_overrides_index_topk(tmp_path):
    base = _config()
    stage = CurriculumStage(
        seq_len=8192,
        tokenized_path="stage8k.npz",
        until_step=100,
        index_topk=2048,
    )
    config = _stage_config(
        base, stage, out_dir=tmp_path, index=2, resume="prev/final", schedule_steps=100
    )
    assert config.model.index_topk == 2048
    # A new stage has its own dataset; do not skip its batches using the
    # previous stage's data counter.
    assert config.train.resume_data is False


def test_curriculum_same_stage_resume_skips_consumed_batches(tmp_path):
    base = _config()
    stage = CurriculumStage(seq_len=4096, tokenized_path="stage.npz", until_step=100)
    own = tmp_path / "stage_01_seq4096" / "step_000400"
    config = _stage_config(
        base,
        stage,
        out_dir=tmp_path,
        index=1,
        resume=str(own),
        schedule_steps=1000,
        resume_data=True,
    )
    assert config.train.resume_data is True
    assert config.train.resume == str(own)


def test_curriculum_resume_data_flag_tracks_checkpoint_origin(tmp_path, monkeypatch):
    """Same-stage resume replays the order; a stage hand-off must not skip."""
    import json

    import minideepseek.train.run as train_run
    from minideepseek.config import dump_config
    from minideepseek.train.curriculum import CurriculumSettings, run_curriculum

    base = _config()
    dump_config(base, tmp_path / "base.yaml")
    stages = [
        CurriculumStage(seq_len=2048, tokenized_path="a.npz", until_step=10),
        CurriculumStage(seq_len=4096, tokenized_path="b.npz", until_step=20),
    ]
    out_dir = tmp_path / "out"
    own = out_dir / "stage_00_seq2048" / "step_000010"
    own.mkdir(parents=True)
    (own / "meta.json").write_text(json.dumps({"step": 10}), encoding="utf-8")

    seen = []

    def fake_run(config, mode):
        seen.append(config)
        return {"checkpoint": str(own), "summary": {"interrupted": False}}

    monkeypatch.setattr(train_run, "run", fake_run)
    reports = run_curriculum(
        CurriculumSettings(
            base_config=str(tmp_path / "base.yaml"), stages=stages, out_dir=str(out_dir)
        )
    )

    assert len(reports) == 2
    assert seen[0].train.resume == str(own)
    assert seen[0].train.resume_data is True
    assert seen[1].train.resume == str(own)
    assert seen[1].train.resume_data is False


def test_curriculum_stage_index_topk_window(tmp_path):
    base = _config()
    stage = CurriculumStage(
        seq_len=8192,
        tokenized_path="stage8k.npz",
        until_step=1000,
        index_topk=2048,
        index_topk_final=64,
        index_topk_warmup_ratio=0.10,
    )
    validate_stages([stage])
    config = _stage_config(
        base,
        stage,
        out_dir=tmp_path,
        index=2,
        resume=None,
        schedule_steps=1000,
        previous_step=800,
    )
    assert config.train.index_topk_final == 64
    # The stage owns the shrink: the legacy end-of-run ratio is disabled.
    assert config.train.index_topk_warmup_ratio == 0.0
    assert config.train.index_topk_start_step == 800
    assert config.train.index_topk_end_step == 820


def test_curriculum_stage_index_topk_window_needs_final():
    stage = CurriculumStage(
        seq_len=8192,
        tokenized_path="stage8k.npz",
        until_step=10,
        index_topk_warmup_ratio=0.1,
    )
    with pytest.raises(ValueError, match="needs index_topk_final"):
        validate_stages([stage])


def test_curriculum_stage_sources_replace_tokenized_path(tmp_path):
    base = _config()
    stage = CurriculumStage(
        seq_len=4096,
        until_step=20,
        sources=[
            {"path": "g.npz", "weight": 0.82, "domain": "general"},
            {"path": "c.npz", "weight": 0.15, "domain": "code"},
            {"path": "t.npz", "weight": 0.03, "domain": "tool"},
        ],
    )
    validate_stages([stage])
    config = _stage_config(base, stage, out_dir=tmp_path, index=0, resume=None, schedule_steps=20)
    assert config.data.tokenized_path is None
    assert [(source.domain, source.weight) for source in config.data.sources] == [
        ("general", 0.82),
        ("code", 0.15),
        ("tool", 0.03),
    ]
    assert config.data.lockstep is True


def test_curriculum_stage_sources_weights_must_sum_to_one():
    stage = CurriculumStage(
        seq_len=4096,
        until_step=10,
        sources=[{"path": "g.npz", "weight": 0.5, "domain": "general"}],
    )
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        validate_stages([stage])


def test_curriculum_stage_val_sources(tmp_path):
    base = _config()
    stage = CurriculumStage(
        seq_len=4096,
        tokenized_path="b.npz",
        until_step=20,
        val_sources=[
            {"path": "vg.npz", "weight": 1.0, "domain": "general"},
            {"path": "vc.npz", "weight": 1.0, "domain": "code"},
        ],
    )
    validate_stages([stage])
    config = _stage_config(base, stage, out_dir=tmp_path, index=0, resume=None, schedule_steps=20)
    assert [source.domain for source in config.train.val_sources] == [
        "general",
        "code",
    ]
    assert config.train.val_sources[0].path == "vg.npz"


def test_curriculum_stage_grad_accum_override(tmp_path):
    base = _config()
    stage = CurriculumStage(seq_len=8192, tokenized_path="stage.npz", until_step=100, grad_accum=8)
    config = _stage_config(base, stage, out_dir=tmp_path, index=2, resume=None, schedule_steps=100)
    assert config.train.grad_accum == 8


def test_curriculum_stage_grad_accum_validation():
    with pytest.raises(ValueError, match="grad_accum must be positive"):
        validate_stages(
            [CurriculumStage(seq_len=2048, tokenized_path="a.npz", until_step=1, grad_accum=0)]
        )
