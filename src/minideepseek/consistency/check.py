"""Consistency gate CLI.

Examples::

    python -m minideepseek.consistency.check --flavor debugmodel --demo
    python -m minideepseek.consistency.check --flavor debugmodel \
        --checkpoint checkpoints/pretrain/final --prompts data/eval/prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import resolve_device
from ..data.io import iter_jsonl
from ..model.checkpoint import load_checkpoint
from ..model.wrapper import build_model
from ..tokenizer import load_tokenizer
from .checks import run_all_checks


def _demo_prompts(tokenizer, count: int = 3) -> list[list[int]]:
    prompts = []
    for index in range(count):
        prompt = [
            {"role": "user", "content": f"请计算 {index + 1} + {index + 2}，并给出最终答案。"}
        ]
        text = tokenizer.render_chat(prompt, add_generation_prompt=True)
        ids = tokenizer.encode(text)
        ids = [value if value < 2048 else value % 2048 for value in ids]
        prompts.append(ids)
    return prompts


def _load_prompts(path: str, tokenizer, flavor: str) -> list[list[int]]:
    prompts = []
    for record in iter_jsonl(path):
        messages = record.get("prompt") or record.get("messages")
        if messages:
            text = tokenizer.render_chat(messages, add_generation_prompt=True)
        else:
            text = str(record.get("text", ""))
        ids = tokenizer.encode(text)
        if flavor in ("debugmodel", "debugmodel_dense"):
            ids = [value % 2048 for value in ids]
        if ids:
            prompts.append(ids)
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", default="debugmodel")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-mtp-layers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompts")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--logprob-threshold", type=float, default=1e-3)
    parser.add_argument("--routing-threshold", type=float, default=0.999)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    tokenizer = load_tokenizer(vocab_limit=2048 if args.flavor.startswith('debugmodel') else None)
    model = build_model(
        args.flavor,
        seq_len=args.seq_len,
        n_mtp_layers=args.n_mtp_layers,
        device=resolve_device(args.device),
        swiglu_limit=10.0,
    )
    if args.checkpoint:
        meta = load_checkpoint(args.checkpoint, model)
        expected = meta.get("tokenizer_hash")
        if expected and expected != tokenizer.sha256:
            raise SystemExit("checkpoint tokenizer hash does not match")
    if args.prompts:
        prompts = _load_prompts(args.prompts, tokenizer, args.flavor)
    else:
        prompts = _demo_prompts(tokenizer)
    report = run_all_checks(
        model,
        tokenizer,
        prompts,
        flavor=args.flavor,
        seq_len=args.seq_len,
        n_mtp_layers=args.n_mtp_layers,
        max_new_tokens=args.max_new_tokens,
        logprob_threshold=args.logprob_threshold,
        routing_threshold=args.routing_threshold,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
