#!/usr/bin/env python3
"""Run lm-evaluation-harness against an exported miniDeepSeek HF directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minideepseek.eval.lm_eval import run_lm_eval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--tasks", required=True, help="comma-separated, e.g. mmlu,arc_easy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results = run_lm_eval(
        args.hf_model,
        [task.strip() for task in args.tasks.split(",") if task.strip()],
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        device=args.device,
        output_path=args.out,
    )
    print(json.dumps(results.get("results", results), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
