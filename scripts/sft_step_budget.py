#!/usr/bin/env python3
"""Measure SFT target-token throughput and recommend a step count.

The response-only mask makes input tokens a bad proxy for loss tokens, and
``data.sources`` weights are loss-token weights after the 2026-11-11 fix.  This
script walks the real mixer for a few steps, counts input/target tokens per
source, and prints what ``steps`` reaches the requested number of epochs over
the primary domain.

    .venv/bin/python scripts/sft_step_budget.py --config configs/sft/general.yaml --epochs 1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=20, help="batches to sample")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--primary", default=None, help="domain to size the budget for")
    args = parser.parse_args()

    from minideepseek.config import TrainConfig, load_config
    from minideepseek.train.run import load_dataset

    config = load_config(TrainConfig, args.config)
    dataset, factory = load_dataset(config, "sft")
    meta_sources = list(dataset.meta.get("sources") or [])
    if not meta_sources:
        raise SystemExit("config has no data.sources; use the single-file budget")
    primary = args.primary
    if primary is None:
        primary = max(meta_sources, key=lambda item: float(item.get("weight") or 0)).get(
            "domain"
        )
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"batches": 0, "input": 0, "target": 0}
    )
    iterator = factory()
    for _ in range(max(int(args.steps), 1)):
        batch = next(iterator)
        source = str(getattr(iterator, "current_source", "unknown"))
        totals[source]["batches"] += 1
        totals[source]["input"] += int(batch.num_tokens)
        totals[source]["target"] += int(batch.target_mask.sum())

    print(f"config: {args.config}")
    print(f"primary: {primary}  epochs: {args.epochs}  sampled batches: {args.steps}")
    print(f"{'source':<16}{'batches':>8}{'in/step':>12}{'target/step':>14}{'target frac':>14}")
    total_target = sum(item["target"] for item in totals.values()) or 1
    for source, item in sorted(totals.items()):
        batches = max(item["batches"], 1)
        print(
            f"{source:<16}{item['batches']:>8}"
            f"{item['input'] / batches:>12.0f}{item['target'] / batches:>14.0f}"
            f"{item['target'] / total_target:>14.3f}"
        )

    primary_entry = next(
        (
            item
            for item in meta_sources
            if str(item.get("domain") or Path(str(item.get("path"))).stem) == str(primary)
            or str(primary) in str(item.get("path"))
        ),
        None,
    )
    if primary_entry is None:
        raise SystemExit(f"cannot find primary source {primary!r} in data.sources")
    primary_name = str(
        primary_entry.get("domain") or Path(str(primary_entry.get("path"))).stem
    )
    primary_meta = next(
        (
            meta
            for source_meta, meta in zip(meta_sources, dataset.datasets, strict=False)
            if str(source_meta.get("domain") or Path(str(source_meta.get("path"))).stem)
            == primary_name
        ),
        None,
    )
    if primary_meta is None:
        raise SystemExit("cannot map the primary source to its dataset")
    target_tokens = int(getattr(primary_meta, "num_target_tokens", 0))
    input_tokens = int(getattr(primary_meta, "num_tokens", 0))
    primary_target_per_step = next(
        (
            item["target"] / max(item["batches"], 1)
            for source, item in totals.items()
            if source == primary_name
        ),
        0.0,
    )
    if primary_target_per_step <= 0:
        print("primary source produced no batches in the sample; increase --steps")
        return 1
    recommended = int(round(target_tokens * args.epochs / primary_target_per_step))
    input_recommended = int(
        round(input_tokens * args.epochs / (totals[primary_name]["input"] / max(totals[primary_name]["batches"], 1)))
    )
    print(
        json.dumps(
            {
                "primary_target_tokens": target_tokens,
                "primary_input_tokens": input_tokens,
                "primary_target_per_step": round(primary_target_per_step, 1),
                "recommended_steps_target_epoch": recommended,
                "recommended_steps_input_epoch": input_recommended,
                "current_steps": int(config.train.steps),
                "target_epochs_at_current_steps": round(
                    config.train.steps * primary_target_per_step / max(target_tokens, 1), 3
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
