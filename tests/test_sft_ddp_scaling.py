"""Regression: SFT/pretrain gradient scaling uses the *global* token count.

The 2-process child script deliberately gives the two ranks different target
token counts; with the old per-rank ``1/target_tokens`` scaling they diverged
after one optimizer step.  The child asserts bit-identical weights.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dist_mode", ["ddp", "fsdp2"])
def test_sft_step_is_rank_invariant(dist_mode: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "test_sft_ddp_scaling.py"),
            "--dist-mode",
            dist_mode,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        check=False,
    )
    assert completed.returncode == 0, (
        f"child failed ({dist_mode}):\n"
        f"stdout:\n{completed.stdout[-4000:]}\n"
        f"stderr:\n{completed.stderr[-4000:]}"
    )
    assert f"PASS sft-{dist_mode}" in completed.stdout
    assert "max weight diff = 0.000e+00" in completed.stdout
