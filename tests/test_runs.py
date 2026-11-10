"""Run identity: per-launch directories and the resume lineage."""

from __future__ import annotations

import json
from pathlib import Path

from minideepseek import runs


def _make_run(out_dir: Path, run_id: str, steps: list[int], *, started_at: str) -> Path:
    path = out_dir / "runs" / run_id
    path.mkdir(parents=True)
    runs.write_meta(path, {"run_id": run_id, "started_at": started_at})
    with (path / runs.METRICS_FILENAME).open("w", encoding="utf-8") as handle:
        for step in steps:
            handle.write(json.dumps({"step": step, "loss": 1.0, "run_id": run_id}) + "\n")
    return path


def test_run_id_carries_time_model_and_world_size():
    run_id = runs.make_run_id("mini_145m_a60m", 145_841_946, 1, stamp="20261110T121245Z")
    assert run_id == "20261110T121245Z_mini_145m_a60m_145.8M_1g"
    assert runs.is_stamp(run_id.split("_")[0])


def test_run_id_uses_launcher_stamp(monkeypatch):
    monkeypatch.setenv(runs.RUN_ID_ENV, "20260101T000000Z")
    assert runs.make_run_id("debugmodel", 12_980_000, 4).startswith("20260101T000000Z_")
    assert runs.make_run_id("debugmodel", 12_980_000, 4).endswith("_4g")


def test_allocate_run_dir_never_reuses_a_directory(tmp_path):
    first = runs.allocate_run_dir(tmp_path, "20260101T000000Z_m_1.0M_1g")
    second = runs.allocate_run_dir(tmp_path, "20260101T000000Z_m_1.0M_1g")
    assert first != second
    assert second.name.endswith("-2")


def test_restarts_do_not_share_a_metrics_file(tmp_path):
    """The bug this replaces: two launches appended to one metrics.jsonl."""
    out_dir = tmp_path / "stage"
    _make_run(out_dir, "20260101T000000Z_a_1.0M_1g", [0, 10, 20], started_at="2026-01-01T00:00:00")
    _make_run(out_dir, "20260101T010000Z_a_1.0M_1g", [0, 10, 20], started_at="2026-01-01T01:00:00")

    infos = runs.list_runs(out_dir)
    assert [info.run_id for info in infos] == [
        "20260101T000000Z_a_1.0M_1g",
        "20260101T010000Z_a_1.0M_1g",
    ]
    for info in infos:
        steps = [record["step"] for record in runs.load_records(info.path / runs.METRICS_FILENAME)]
        assert steps == [0, 10, 20]  # each file is its own launch, no 30->0 rollback

    # A restart at step 0 is a dead end: it must not be spliced into the chain.
    assert [info.run_id for info in runs.lineage(out_dir)] == ["20260101T010000Z_a_1.0M_1g"]


def test_lineage_follows_a_resume_chain(tmp_path):
    out_dir = tmp_path / "stage"
    _make_run(out_dir, "20260101T000000Z_a_1.0M_1g", [0, 10, 20], started_at="2026-01-01T00:00:00")
    # Resumed from step 20.
    _make_run(out_dir, "20260101T010000Z_a_1.0M_1g", [20, 30, 40], started_at="2026-01-01T01:00:00")
    # Resumed from step 40.
    _make_run(out_dir, "20260101T020000Z_a_1.0M_1g", [40, 50], started_at="2026-01-01T02:00:00")
    # An abandoned branch that restarted at 0 and got further than 10.
    _make_run(out_dir, "20260101T030000Z_a_1.0M_1g", [0, 10], started_at="2026-01-01T03:00:00")

    chain = runs.lineage(out_dir, "20260101T020000Z_a_1.0M_1g")
    assert [info.run_id for info in chain] == [
        "20260101T000000Z_a_1.0M_1g",
        "20260101T010000Z_a_1.0M_1g",
        "20260101T020000Z_a_1.0M_1g",
    ]
    records = runs.collect_records(chain)
    assert [record["step"] for record in records] == [0, 10, 20, 20, 30, 40, 40, 50]


def test_latest_run_ignores_directories_without_records(tmp_path):
    out_dir = tmp_path / "stage"
    _make_run(out_dir, "20260101T000000Z_a_1.0M_1g", [0, 10], started_at="2026-01-01T00:00:00")
    empty = out_dir / "runs" / "20260101T050000Z_a_1.0M_1g"
    empty.mkdir(parents=True)
    runs.write_meta(empty, {"run_id": empty.name, "started_at": "2026-01-01T05:00:00"})

    assert runs.latest_run(out_dir).run_id == "20260101T000000Z_a_1.0M_1g"
    assert len(runs.list_runs(out_dir)) == 2


def test_find_run_accepts_a_unique_prefix(tmp_path):
    out_dir = tmp_path / "stage"
    _make_run(out_dir, "20260101T000000Z_a_1.0M_1g", [0], started_at="2026-01-01T00:00:00")
    _make_run(out_dir, "20260102T000000Z_a_1.0M_1g", [0], started_at="2026-01-02T00:00:00")

    assert runs.find_run(out_dir, "20260102").run_id.startswith("20260102")
    assert runs.find_run(out_dir, "2026") is None  # ambiguous
    assert runs.find_run(out_dir, "nope") is None


def test_update_meta_merges_resume_provenance(tmp_path):
    path = _make_run(tmp_path, "20260101T000000Z_a_1.0M_1g", [0], started_at="2026-01-01T00:00:00")
    runs.update_meta(path, resumed_from="/x/final", resumed_at_step=20)
    meta = runs.read_meta(path)
    assert meta["resumed_from"] == "/x/final"
    assert meta["resumed_at_step"] == 20
    assert meta["run_id"] == "20260101T000000Z_a_1.0M_1g"


def test_legacy_flat_metrics_is_still_detected(tmp_path):
    (tmp_path / runs.METRICS_FILENAME).write_text('{"step": 0, "loss": 1.0}\n', encoding="utf-8")
    assert runs.list_runs(tmp_path) == []
    assert runs.legacy_metrics_path(tmp_path) is not None


def test_continuation_needs_a_checkpoint_and_the_same_flavour(tmp_path):
    """A resume appends to the live run; a different model starts a new one."""
    out_dir = tmp_path / "stage"
    runs.write_current(
        out_dir, run_id="20260101T000000Z_m_1.0M_1g", flavor="m", total_params=1e6
    )
    assert runs.continuation(out_dir, "m") is None  # nothing to resume yet

    (out_dir / "step_000001").mkdir(parents=True)
    assert runs.continuation(out_dir, "m") == "20260101T000000Z_m_1.0M_1g"
    assert runs.continuation(out_dir, "other-model") is None


def test_campaign_stamp_follows_resumable_state(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    out = tmp_path / "campaign"
    (out / "stage_00_seq2048" / "step_000100").mkdir(parents=True)
    first = runs.campaign_stamp(out, logs, "curriculum_x")
    assert runs.is_stamp(first)
    assert runs.campaign_stamp(out, logs, "curriculum_x") == first

    # ``final`` counts as resumable too (a restarted finished stage resumes it).
    (out / "stage_00_seq2048" / "final").mkdir()
    assert runs.campaign_stamp(out, logs, "curriculum_x") == first

    # Nothing left to resume: the next launch is a new training -> new stamp.
    for child in (out / "stage_00_seq2048").iterdir():
        child.rmdir()
    monkeypatch.setattr(runs, "utc_stamp", lambda: "20270101T000000Z")
    assert runs.campaign_stamp(out, logs, "curriculum_x") == "20270101T000000Z"


def test_launch_log_path_puts_the_stamp_last(tmp_path):
    config = tmp_path / "curriculum_145m_g2.yaml"
    config.write_text(f"out_dir: {tmp_path / 'cp'}\n", encoding="utf-8")
    path = runs.launch_log_path(config, tmp_path / "logs")
    assert path.name.startswith("curriculum_145m_g2_")
    assert runs.is_stamp(path.stem.rsplit("_", 1)[1])
