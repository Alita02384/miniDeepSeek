#!/usr/bin/env python3
"""Smoke-check the local InternLM2 reward model.

    python scripts/check_reward_model.py --path internlm2-1_8b-reward
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minideepseek.rl.reward_model import LocalRewardModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="internlm2-1_8b-reward")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    parser.add_argument("--response", default="你好，我是一个语言模型。")
    args = parser.parse_args()
    model = LocalRewardModel(args.path, dtype=args.dtype)
    score = model.get_score([{"role": "user", "content": args.prompt}], args.response)
    print(f"{Path(args.path).resolve()} reward={score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
