#!/usr/bin/env python3
"""Run a multi-stage sequence-length curriculum.

    .venv/bin/python scripts/train_curriculum.py \
        --config configs/pretrain/length_curriculum.yaml

Stage datasets must be prepared separately (2K/4K/8K `.npz` + `.json`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Before torch: the Triton/inductor caches must point at the repo-local,
# persistent directory or every restarted box recompiles the attention kernels.
from minideepseek.paths import configure_compile_cache  # noqa: E402

configure_compile_cache()

from minideepseek.config import load_config  # noqa: E402
from minideepseek.train.curriculum import CurriculumSettings, run_curriculum  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    settings = load_config(CurriculumSettings, args.config)
    reports = run_curriculum(settings)
    print(json.dumps(reports, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
