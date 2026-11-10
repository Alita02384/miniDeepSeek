from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prewarm_attn.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_prewarm_dry_run_plans_one_entry_per_sequence_length():
    completed = _run("--flavor", "mini_145m_a60m", "--seq-len", "2048", "4096", "--dry-run")

    assert completed.returncode == 0, completed.stderr
    plan = [line for line in completed.stdout.splitlines() if line.startswith("[prewarm] seq")]
    assert len(plan) == 2
    assert "seq=2048" in plan[0] and "topk=512" in plan[0]
    assert "seq=4096" in plan[1] and "topk=1024" in plan[1]
    # 145M hybrid map: two SWA layers, three CSA (ratio 4), two HCA (ratio 32).
    assert "fused_layers=[2, 3, 4, 5, 6]" in plan[0]


def test_prewarm_refuses_to_tune_on_cpu():
    completed = _run("--flavor", "debugmodel", "--seq-len", "128")

    assert completed.returncode == 2
    assert "no CUDA device" in completed.stderr


def test_prewarm_max_seq_len_skips_longer_stages():
    completed = _run(
        "--flavor",
        "mini_145m_a60m",
        "--seq-len",
        "2048",
        "4096",
        "--max-seq-len",
        "2048",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert "seq4096: skipped" in completed.stdout
    assert "seq=4096" not in completed.stdout
    assert "seq=2048" in completed.stdout


def test_prewarm_max_seq_len_that_skips_everything_fails_loudly():
    completed = _run(
        "--flavor",
        "mini_145m_a60m",
        "--seq-len",
        "2048",
        "--max-seq-len",
        "16",
        "--dry-run",
    )

    assert completed.returncode != 0
    assert "skips every stage" in completed.stderr
