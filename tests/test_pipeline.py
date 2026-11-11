"""Pipeline planning helpers: budgets, done detection and preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from minideepseek.pipeline import (
    _dataset_exists,
    _preflight,
    _pretrain_tokens,
    _rl_stage,
    _sft_stage,
    _sft_step_plan,
    _stage_done,
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
    _meta(tmp_path / "sft_tool.npz", tokens=5_000, target_tokens=2_000)
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


def test_sft_stage_honors_explicit_steps(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    general = _meta(data / "sft_general.npz", tokens=1_000_000, target_tokens=800_000)
    config = tmp_path / "sft.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "batch_tokens": 1000,
                    "sources": [
                        {"path": str(general), "weight": 1.0, "domain": "general"}
                    ],
                },
                "train": {"grad_accum": 1, "resume": "base"},
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    _sft_stage(
        {
            "name": "general",
            "config": str(config),
            "steps": 7,
            "resume": "base",
            "out_dir": "out",
        },
        run_dir=run_dir,
        nproc=1,
        tokenization_dir=data,
        dry_run=True,
    )
    generated = yaml.safe_load(
        (run_dir / "configs" / "sft_general.yaml").read_text(encoding="utf-8")
    )
    assert generated["train"]["steps"] == 7
    # The planned budget would have been 1000 steps (800 target tokens per
    # 1000-token micro-batch); the stage override wins.
    assert _sft_step_plan(
        yaml.safe_load(config.read_text(encoding="utf-8")),
        stage_name="general",
        epochs=1.0,
        tokenization_dir=data,
    )[0] == 1000


def test_rl_stage_overrides_steps_and_teachers(tmp_path):
    config = tmp_path / "rl.yaml"
    config.write_text(
        yaml.safe_dump({"teachers": [{"name": "old", "key": "old", "checkpoint": "old"}]}),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    _rl_stage(
        {
            "id": "grpo_general",
            "kind": "grpo",
            "config": str(config),
            "data": "grpo.jsonl",
            "resume": "sft",
            "out_dir": "grpo_out",
            "steps": 3,
        },
        run_dir=run_dir,
        nproc=1,
        dry_run=True,
    )
    generated = yaml.safe_load(
        (run_dir / "configs" / "grpo_general.yaml").read_text(encoding="utf-8")
    )
    assert generated["train"]["steps"] == 3
    assert generated["train"]["out_dir"] == "grpo_out"

    teacher = {"name": "new", "key": "new", "checkpoint": "new/final"}
    _rl_stage(
        {
            "id": "opd",
            "kind": "opd",
            "config": str(config),
            "data": "opd.jsonl",
            "resume": "sft",
            "out_dir": "opd_out",
            "teachers": [teacher],
        },
        run_dir=run_dir,
        nproc=1,
        dry_run=True,
    )
    generated = yaml.safe_load((run_dir / "configs" / "opd.yaml").read_text(encoding="utf-8"))
    assert generated["teachers"] == [teacher]


def test_preflight_prefers_stage_teacher_checkpoints(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "opd_prompts_three.jsonl").write_text("{}\n", encoding="utf-8")
    good_teacher = tmp_path / "good_teacher" / "final"
    good_teacher.mkdir(parents=True)
    config = tmp_path / "opd.yaml"
    config.write_text(
        yaml.safe_dump(
            {"teachers": [{"name": "old", "key": "old", "checkpoint": str(tmp_path / "missing")}]}
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
                "teachers": [
                    {"name": "new", "key": "new", "checkpoint": str(good_teacher)}
                ],
            }
        ],
    }
    # The config's teacher is missing, but the stage override points at a real
    # checkpoint -- preflight must validate what will actually be loaded.
    _preflight(spec, data, dry_run=True)


def test_preflight_allows_outputs_of_earlier_stages(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "grpo_general.jsonl").write_text("{}\n", encoding="utf-8")
    (data / "opd_prompts_three.jsonl").write_text("{}\n", encoding="utf-8")
    base = tmp_path / "base" / "final"
    base.mkdir(parents=True)
    sft_out = tmp_path / "e2e" / "sft_general"
    grpo_out = tmp_path / "e2e" / "grpo_general"
    rl_config = tmp_path / "rl.yaml"
    rl_config.write_text(
        yaml.safe_dump({"teachers": [{"name": "t", "key": "t", "checkpoint": "unused"}]}),
        encoding="utf-8",
    )
    sft_config = tmp_path / "sft.yaml"
    sft_config.write_text(yaml.safe_dump({"data": {}, "train": {}}), encoding="utf-8")
    (tmp_path / "tokenization").mkdir()
    (tmp_path / "tokenization" / "sft_general.npz").write_bytes(b"")
    spec = {
        "tokenization_dir": str(tmp_path / "tokenization"),
        "stages": [
            {
                "id": "sft_general",
                "kind": "sft",
                "name": "general",
                "config": str(sft_config),
                "resume": str(base),
                "out_dir": str(sft_out),
            },
            {
                "id": "grpo_general",
                "kind": "grpo",
                "config": str(rl_config),
                "data": str(data / "grpo_general.jsonl"),
                "resume": str(sft_out / "final"),
                "out_dir": str(grpo_out),
            },
            {
                "id": "opd",
                "kind": "opd",
                "config": str(rl_config),
                "data": str(data / "opd_prompts_three.jsonl"),
                "resume": str(sft_out / "final"),
                "out_dir": str(tmp_path / "e2e" / "opd"),
                "teachers": [
                    {"name": "g", "key": "g", "checkpoint": str(grpo_out / "final")}
                ],
            },
        ],
    }
    _preflight(spec, data, dry_run=True)


def test_dataset_exists_accepts_sidecars(tmp_path):
    logical = tmp_path / "sft_general.npz"
    assert not _dataset_exists(logical)
    for name in ("input_ids", "positions", "target_mask", "offsets"):
        logical.with_suffix(f".{name}.npy").write_bytes(b"")
    logical.with_suffix(".json").write_text("{}", encoding="utf-8")
    assert _dataset_exists(logical)


def test_stage_done_report_checks_html(tmp_path):
    out = tmp_path / "reports" / "posttrain.html"
    stage = {"id": "report", "kind": "report", "out": str(out)}
    assert not _stage_done(stage, tokenization_dir=tmp_path)
    out.parent.mkdir()
    out.write_text("<html></html>", encoding="utf-8")
    assert _stage_done(stage, tokenization_dir=tmp_path)


def test_preflight_report_does_not_require_run_root(tmp_path):
    spec = {
        "tokenization_dir": str(tmp_path / "tokenization"),
        "stages": [
            {
                "id": "report",
                "kind": "report",
                "root": "checkpoints/does-not-exist-yet",
                "out": str(tmp_path / "report.html"),
            }
        ],
    }
    _preflight(spec, tmp_path, dry_run=True)


def test_dry_run_writes_separate_state(tmp_path):
    config = tmp_path / "pipeline.yaml"
    run_dir = tmp_path / "run"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "dry",
                "nproc": 1,
                "run_dir": str(run_dir),
                "stages": [
                    {
                        "id": "posttrain_report",
                        "kind": "report",
                        "root": str(tmp_path / "checkpoints"),
                        "out": str(tmp_path / "report.html"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    from minideepseek.pipeline import run_pipeline

    run_pipeline(config, dry_run=True)
    assert (run_dir / "state_dry_run.json").exists()
    assert not (run_dir / "state.json").exists()
