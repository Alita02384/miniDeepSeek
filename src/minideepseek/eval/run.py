"""Evaluation CLI.

    python -m minideepseek.eval.run --config configs/eval/math.yaml
"""

from __future__ import annotations

import argparse
import json

from ..config import EvalConfig, load_config
from .evaluate import evaluate, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    config = load_config(EvalConfig, args.config)
    report = evaluate(config)
    if args.out or config.out_path:
        write_report(report, args.out or config.out_path)
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
