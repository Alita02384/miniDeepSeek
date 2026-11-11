"""The metrics consolidator: which source wins, and what it recovers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CONSOLE_0 = (
    "step      0 | loss 12.0116 (ema 12.0116) | main 9.2386 | mtp 9.2373 | "
    "lr 2.00e-06 | |g| 3.221 | 6.30k tok/s | 11.0 GiB peak | ETA 4.2d"
)
CONSOLE_10 = (
    "step     10 | loss 11.9252 (ema 12.0038) | main 9.1565 | lr 2.20e-05 | "
    "|g| 3.329 | 9.43k tok/s | ETA 3.0d"
)


@pytest.fixture()
def merger():
    spec = importlib.util.spec_from_file_location(
        "merge_metrics", ROOT / "scripts" / "merge_metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layout(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "out" / "stage_01_seq4096" / "runs" / "runA"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"step": 90, "loss": 5.0, "main_loss": 4.0}),
                json.dumps({"step": 100, "loss": 3.5, "main_loss": 2.8, "lr": 6e-4}),
                "not json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "curriculum.log").write_text(
        "\n".join([CONSOLE_0, CONSOLE_10, "step     90 | loss 5.1 | main 4.1"]) + "\n",
        encoding="utf-8",
    )
    return tmp_path / "out", logs


def test_console_lines_are_parsed_into_metric_records(merger):
    record = merger.parse_console_line(CONSOLE_0)
    assert record["step"] == 0
    assert record["loss"] == 12.0116
    assert record["loss_ema"] == 12.0116
    assert record["main_loss"] == 9.2386
    assert record["lr"] == 2e-06
    assert record["grad_norm"] == 3.221
    # "6.30k tok/s" is 6300, not 6.3
    assert record["tokens_per_sec"] == 6300.0
    assert record["mem_peak_mb"] == pytest.approx(11.0 * 1024)
    assert record["eta_seconds"] == pytest.approx(4.2 * 86400)
    assert merger.parse_console_line("not a step line") is None


def test_structured_records_beat_the_console_echo(merger, tmp_path):
    out, logs = _layout(tmp_path)
    records, provenance = merger.collect(out, logs)

    assert sorted(records) == [0, 10, 90, 100]
    # 0 and 10 exist only in the console log: they are the pre-metrics head
    assert records[0]["source"] == "console"
    assert records[10]["source"] == "console"
    # step 90 appears in both; the structured row must win
    assert records[90]["source"] == "metrics"
    assert records[90]["loss"] == 5.0
    # the stage is inferred from the directory, not guessed from the log name
    assert records[100]["stage"] == 1
    assert records[100]["seq_len"] == 4096
    assert records[0]["stage"] is None
    assert {item["kind"] for item in provenance} == {"metrics", "console"}


def test_coverage_reports_gaps_and_console_only_steps(merger, tmp_path):
    out, logs = _layout(tmp_path)
    stats = merger.coverage(merger.collect(out, logs)[0])
    assert stats["first"] == 0 and stats["last"] == 100
    assert stats["console_only"] == [0, 10]
    assert (11, 89) in stats["gaps"]


def test_write_emits_sorted_jsonl(merger, tmp_path):
    out, logs = _layout(tmp_path)
    assert merger.main([str(out), "--logs-dir", str(logs), "--write"]) == 0
    written = (out / "metrics_all.jsonl").read_text(encoding="utf-8").splitlines()
    steps = [json.loads(line)["step"] for line in written]
    assert steps == sorted(steps) == [0, 10, 90, 100]
