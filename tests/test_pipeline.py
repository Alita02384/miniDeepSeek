"""Pipeline planning helpers: budgets, done detection and preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from minideepseek.pipeline import (
    _pretrain_tokens,
    _preflight,
    _sft_step_plan,
    _stage_done,
    _checkpoint_done,
)


def _meta(path: Path, *, tokens: int, target_tokens: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(
        json.dumps({"tokens": tokens, "target_tokens": target_tokens}),
        encoding="utf-8",
    )
    return path


def test_sft_step_plan_uses_all_sources_not_one_file(tmp_path):
    general = _meta(tmp_path / "sft_general.npz", tokens=1_000_000, target_tokens=860_000)
    code = _meta(tmp_path / "sft_code.npz", tokens=400_000, target_tokens=150_000)
    config = {
        "data": {
            "batch_tokens": 8192,
            "sources": [
                {"path": str(general), "weight": 0.9, "domain": "general"},
                {"path": str(code), "weight": 0.1, "domain": "code"},
            ],
        },
        "train": {"grad_accum": 16},
    }
    steps, primary_targets, per_step = _sft_step_plan(
        config, stage_name="general", epochs=1.0, tokenization_dir=tmp_path
    )
    assert primary_targets == 860_000
    assert steps > 1
    assert per_step > 0
    # The old code read one sft_general.npz and ignored the mixture.
    assert steps != 1


def test_sft_step_plan_single_file(tmp_path):
    path = _meta(tmp_path / "sft_tool.npz", tokens=5_000, target_tokens=2_000)
    config = {"data": {"batch_tokens": 1000}, "train": {"grad_accum": 2}}
    steps, primary_targets, _ = _sft_step_plan(
        config, stage_name="tool", epochs=2.0, tokenization_dir=tmp_path
    )
    assert primary_targets == 2_000
    assert steps == 2  # 2000 * 2 / (1000 * 2)


def test_stage_done_curriculum_finds_any_stage(tmp_path):
    out = tmp_path / "curriculum"
    (out / "stage_00_seq2048" / "final").mkdir(parents=True)
    (out / "stage_00_seq2048" / "final" / "meta.json").write_text("{}", encoding="utf-8")
    assert _stage_done({"kind": "curriculum", "out_dir": str(out)}, tokenization_dir=tmp_path)
    assert not _stage_done(
        {"kind": "curriculum", "out_dir": str(tmp_path / "empty")},
        tokenization_dir=tmp_path,
    )


def test_pretrain_tokens_sums_domains(tmp_path):
    for domain in ("general", "code"):
        _meta(tmp_path / f"pretrain_{domain}_2k.npz", tokens=100, target_tokens=100)
    total, sources = _pretrain_tokens(
        {},
        tokenization_dir=tmp_path,
        domains=["general", "code"],
        seq_len=2048,
        legacy_layout=False,
    )
    assert total == 200
    assert len(sources) == 2


def test_preflight_fails_on_missing_teacher_checkpoint(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "opd_prompts_three.jsonl").write_text("{}\n", encoding="utf-8")
    teacher = tmp_path / "missing_teacher" / "final"
    config = tmp_path / "opd.yaml"
    config.write_text(
        yaml.safe_dump(
            {"teachers": [{"name": "t", "key": "t", "checkpoint": str(teacher)}]}
        ),
        encoding="utf-8",
    )
    spec = {
        "tokenization_dir": str(tmp_path / "tokenization"),
        "stages": [
            {
                "id": "opd",
                "kind": "opd",
                "config": str(config),
                "data": str(data / "opd_prompts_three.jsonl"),
            }
        ],
    }
    with pytest.raises(SystemExit, match="missing paths"):
        _preflight(spec, data, dry_run=True)
