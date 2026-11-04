#!/usr/bin/env python3
"""Export GRPO data + a runnable verl command from miniDeepSeek records.

    .venv/bin/python scripts/export_verl.py \
        --hf-model hf/mini_60m \
        --train data/grpo/math.jsonl \
        --out-dir runs/verl_math

Writes ``train.jsonl`` / ``train.parquet``, ``verl_config.yaml`` (the resolved
values for reference), and ``run.sh`` (flattened Hydra overrides for
``verl.trainer.main_ppo``).  verl is an optional CUDA dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minideepseek.data.grpo import load_records  # noqa: E402
from minideepseek.rl.verl import (  # noqa: E402
    build_verl_config,
    flatten_overrides,
    write_verl_config,
    write_verl_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--rollout", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--experiment", default="minideepseek_grpo")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.train)
    train_path = write_verl_dataset(records, out / "train.jsonl")
    val_records = load_records(args.val) if args.val else records[: min(len(records), 16)]
    val_path = write_verl_dataset(val_records, out / "val.jsonl")

    config = build_verl_config(
        hf_model_path=args.hf_model,
        train_files=[str(train_path)],
        val_files=[str(val_path)],
        output_dir=str(out / "checkpoints"),
        n_gpus_per_node=args.n_gpus,
        total_epochs=args.epochs,
        rollout=args.rollout,
        experiment_name=args.experiment,
    )
    write_verl_config(config, out / "verl_config.yaml")
    command = "python -m verl.trainer.main_ppo \\\n  " + " \\\n  ".join(
        flatten_overrides(config)
    )
    (out / "run.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    (out / "run.sh").chmod(0o755)
    print(json.dumps({"out_dir": str(out), "command": command}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
