#!/usr/bin/env python3
"""Compile the gather-attention kernels *before* step 0 starts the clock.

Triton compiles a kernel the first time a launch shape reaches it, and
attn_gym tunes every candidate it keeps: 18 binaries on the non-TMA path, one
to two minutes each on a consumer GPU.  Each stage adds its own ``index_topk``
specialization, so a fresh instance spends 20-40 minutes on step 0 with no
output at all -- indistinguishable from a hang, and thrown away entirely if the
process is killed.

This script pays that cost once, in one visible pass, and leaves the artifacts
in the persistent cache (``.cache/triton`` plus ``.cache/attn_gym``) that
``configure_compile_cache`` points every entry point at.  The trainer then
starts in seconds, and later stages reuse the same entries.

Examples::

    # every stage of a curriculum (one optimizer step per sequence length)
    python scripts/prewarm_attn.py --curriculum configs/pretrain/generated/curriculum_145m_g1.yaml

    # one shape at a time, matching the training batch of a 4K stage
    python scripts/prewarm_attn.py --flavor mini_145m_a60m --seq-len 4096 --docs 2

    # what would be warmed, without touching a GPU
    python scripts/prewarm_attn.py --flavor mini_145m_a60m --seq-len 2048 --dry-run

Notes:
  * Run it on the same GPU model and the same torch/Triton versions as
    training; the cache key embeds both the architecture and the kernel source.
  * The fused path is forced (``MINIDEEPSEEK_ATTN_BACKEND=attn_gym``) so a
    kernel that cannot compile fails loudly here instead of silently falling
    back to the 8x slower reference path mid-run.
  * ``MINIDEEPSEEK_ATTN_BF16=1`` warms the bf16 variant instead of the default
    fp32 one; warm both by running the script twice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Must run before torch: Triton reads TRITON_CACHE_DIR when it is imported.
from minideepseek.paths import (  # noqa: E402
    COMPILE_CACHE_ENV,
    configure_compile_cache,
)

CACHE_ENV = configure_compile_cache()

import torch  # noqa: E402

from minideepseek.config import (  # noqa: E402
    ModelSettings,
    TrainConfig,
    load_config,
    resolve_device,
)
from minideepseek.model import build_model  # noqa: E402
from minideepseek.train.benchmark import (  # noqa: E402
    run_benchmark,
    synthetic_batch_factory,
)
from minideepseek.train.curriculum import (  # noqa: E402
    CurriculumSettings,
    stage_train_configs,
)
from minideepseek.train.optimizer import build_optimizer  # noqa: E402

#: One distinct gather-attention launch shape: (kernel dtype, heads, head_dim,
#: tokens, compressed tokens, selected top-k, window).
Variant = tuple[str, int, int, int, int, int, int]


def _describe(variant: Variant) -> str:
    dtype, heads, head_dim, tokens, compressed, topk, window = variant
    return f"{dtype} H={heads} D={head_dim} T={tokens} S={compressed} K={topk} win={window}"


class KernelProbe:
    """Time the first launch of every distinct gather-attention variant."""

    def __init__(self, warn_seconds: float) -> None:
        self.warn_seconds = warn_seconds
        self.calls: dict[Variant, int] = {}
        self.first_call_s: dict[Variant, float] = {}
        self.later_s: dict[Variant, list[float]] = {}
        self._original: Any = None

    def install(self) -> None:
        from minideepseek.model import attention

        self._original = attention._fused_gather_attention
        attention._fused_gather_attention = self._call  # type: ignore[assignment]

    def _call(
        self,
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        cu_seqlens,
        cu_seqlens_k,
        sliding_window_size,
        *,
        scale: float,
    ):
        variant: Variant = (
            str(query.dtype).replace("torch.", ""),
            int(query.shape[1]),
            int(query.shape[3]),
            int(query.shape[2]),
            int(sparse_kv.shape[2]),
            int(kv_indices.shape[-1]),
            int(sliding_window_size),
        )
        first = variant not in self.calls
        if first:
            print(f"[prewarm] compiling {_describe(variant)} ...", flush=True)
        start = time.perf_counter()
        output = self._original(
            query,
            local_kv,
            sparse_kv,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            sliding_window_size,
            scale=scale,
        )
        elapsed = time.perf_counter() - start
        self.calls[variant] = self.calls.get(variant, 0) + 1
        if first:
            self.first_call_s[variant] = elapsed
            print(f"[prewarm]   compiled in {elapsed:,.1f}s", flush=True)
            if elapsed > self.warn_seconds:
                print(
                    f"[prewarm]   WARNING: over --warn-seconds={self.warn_seconds:.0f}; "
                    "this variant is the cold-compile outlier",
                    flush=True,
                )
        else:
            self.later_s.setdefault(variant, []).append(elapsed)
        return output

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for variant, compile_s in self.first_call_s.items():
            later = self.later_s.get(variant, [])
            rows.append(
                {
                    "variant": _describe(variant),
                    "dtype": variant[0],
                    "heads": variant[1],
                    "head_dim": variant[2],
                    "tokens": variant[3],
                    "compressed_tokens": variant[4],
                    "topk": variant[5],
                    "window": variant[6],
                    "compile_seconds": round(compile_s, 1),
                    "calls": self.calls[variant],
                    "steady_ms": round(1000.0 * sum(later) / len(later), 2) if later else None,
                }
            )
        return sorted(rows, key=lambda row: row["compile_seconds"], reverse=True)


def _entries(args: argparse.Namespace) -> list[tuple[str, TrainConfig]]:
    """Resolve the (label, config) pairs to warm, one per sequence length."""
    if args.curriculum:
        settings = load_config(CurriculumSettings, args.curriculum)
        configs = stage_train_configs(settings)
        entries = [
            (f"stage_{index:02d}_seq{stage.seq_len}", config)
            for index, (stage, config) in enumerate(zip(settings.stages, configs, strict=True))
        ]
    else:
        config = _single_config(args)
        lengths = args.seq_len or [int(config.model.seq_len)]
        entries = []
        for length in lengths:
            staged = TrainConfig.from_dict(config.to_dict())
            staged.model.seq_len = int(length)
            staged.model.max_context = max(int(staged.model.max_context), int(length))
            entries.append((f"seq{length}", staged))
    if args.index_topk:
        expanded: list[tuple[str, TrainConfig]] = []
        for label, config in entries:
            for topk in args.index_topk:
                variant = TrainConfig.from_dict(config.to_dict())
                variant.model.index_topk = int(topk)
                expanded.append((f"{label}_k{topk}", variant))
        entries = expanded
    return entries


def _single_config(args: argparse.Namespace) -> TrainConfig:
    if args.config:
        return load_config(TrainConfig, args.config)
    if args.model_config:
        return TrainConfig(model=load_config(ModelSettings, args.model_config))
    flavor = args.flavor or "mini_145m_a60m"
    path = ROOT / "configs" / "model" / f"{flavor}.yaml"
    if not path.exists():
        raise SystemExit(f"unknown flavor {flavor!r}; expected {path}")
    return TrainConfig(model=load_config(ModelSettings, path))


def _documents(config: TrainConfig, override: int | None) -> int:
    """Packs per step, defaulting to the stage's own ``batch_tokens`` budget."""
    if override is not None:
        return max(int(override), 1)
    seq_len = int(config.model.seq_len)
    return max(int(config.data.batch_tokens) // max(seq_len, 1), 1)


def _plan(label: str, config: TrainConfig, docs: int) -> dict[str, Any]:
    """Describe one warm-up target: shapes, fused layers, derived top-k."""
    model = config.model
    seq_len = int(model.seq_len)
    compress_ratios = [int(r) for r in model.compress_ratios]
    fused_layers = [index for index, ratio in enumerate(compress_ratios) if ratio > 1]
    topk = model.index_topk
    derived = min(2048, max(seq_len // 4, 64))
    return {
        "label": label,
        "seq_len": seq_len,
        "documents": docs,
        "tokens_per_step": seq_len * docs,
        "dtype": model.dtype,
        "fused_layers": fused_layers,
        "compress_ratios": [compress_ratios[index] for index in fused_layers],
        "index_topk": int(topk) if topk is not None else derived,
    }


def _warm(label: str, config: TrainConfig, args: argparse.Namespace, docs: int) -> dict[str, Any]:
    model_settings = config.model
    device = resolve_device(model_settings.device if args.device == "auto" else args.device)
    if device.type != "cuda":
        raise SystemExit("prewarm needs a CUDA device; nothing to compile on CPU")
    dtype = model_settings.dtype if device.type == "cuda" else "float32"
    seq_len = int(model_settings.seq_len)
    torch.manual_seed(int(model_settings.seed))
    started = time.perf_counter()
    # Mirror the trainer's memory layout: the 8K stage only fits 24 GB with
    # activation checkpointing, and this step would otherwise OOM first.
    model = build_model(
        model_settings,
        device=device,
        dtype=dtype,
        activation_checkpointing=bool(config.train.activation_checkpointing),
    )
    optimizer = build_optimizer(model, config.optim)
    factory = synthetic_batch_factory(int(model_settings.vocab_size), docs, seq_len, device)
    result = run_benchmark(
        model,
        optimizer,
        factory,
        steps=args.steps,
        warmup=1,  # the warm-up step is where every compile lands
        device=device,
        grad_clip=config.optim.grad_clip,
        mtp_weight=args.mtp_weight,
        label=label,
    )
    spent = time.perf_counter() - started
    peak = f"{result.peak_mem_mb / 1024:.1f} GiB peak, " if result.peak_mem_mb else ""
    print(
        f"[prewarm] {label}: steady {result.tokens_per_sec:,.0f} tok/s, {peak}{spent:,.0f}s total",
        flush=True,
    )
    del model, optimizer
    torch.cuda.empty_cache()
    return {
        "label": label,
        "seq_len": seq_len,
        "documents": docs,
        "tokens_per_sec": round(result.tokens_per_sec, 1),
        "peak_mem_mb": result.peak_mem_mb,
        "total_seconds": round(spent, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--curriculum", default=None, help="curriculum YAML: one warm-up per stage")
    source.add_argument("--config", default=None, help="train YAML")
    source.add_argument("--model-config", default=None, help="model YAML")
    source.add_argument("--flavor", default=None, help="bundled model flavor")
    parser.add_argument(
        "--seq-len", type=int, nargs="+", default=None, help="overrides --config/--flavor"
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=None,
        help="packs per step; defaults to data.batch_tokens / seq_len, "
        "i.e. the micro-batch the trainer will build",
    )
    parser.add_argument(
        "--index-topk",
        type=int,
        nargs="+",
        default=None,
        help="warm these top-k values instead of the model's own; an active shrink "
        "schedule visits a power-of-two ladder and each rung is its own kernel",
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="measured steps after the warm-up step"
    )
    parser.add_argument("--mtp-weight", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warn-seconds", type=float, default=180.0)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="skip stages longer than this (e.g. 4096 when 8K may not fit the card)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first stage that fails instead of warming the rest first",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    if args.seq_len and args.curriculum:
        parser.error(
            "--seq-len overrides --config/--flavor; a --curriculum carries its own lengths"
        )

    entries = _entries(args)
    if args.max_seq_len is not None:
        skipped = [
            label for label, config in entries if int(config.model.seq_len) > args.max_seq_len
        ]
        for label in skipped:
            print(f"[prewarm] {label}: skipped (seq_len > --max-seq-len={args.max_seq_len})")
        entries = [entry for entry in entries if int(entry[1].model.seq_len) <= args.max_seq_len]
        if not entries:
            raise SystemExit(f"--max-seq-len={args.max_seq_len} skips every stage")
    planned = [(label, config, _documents(config, args.docs)) for label, config in entries]
    for label, config, docs in planned:
        plan = _plan(label, config, docs)
        print(
            f"[prewarm] {label}: seq={plan['seq_len']} docs={plan['documents']} "
            f"dtype={plan['dtype']} fused_layers={plan['fused_layers']} "
            f"topk={plan['index_topk']}",
            flush=True,
        )
    if args.dry_run:
        return 0
    if not torch.cuda.is_available():
        parser.error(
            "no CUDA device; there is nothing to compile (use --dry-run to inspect the plan)"
        )

    # A missing kernel must fail here, not degrade the trainer by a factor of 8.
    os.environ.setdefault("MINIDEEPSEEK_ATTN_BACKEND", "attn_gym")
    probe = KernelProbe(args.warn_seconds)
    probe.install()

    print(
        "[prewarm] cache: " + " ".join(f"{name}={CACHE_ENV[name]}" for name in COMPILE_CACHE_ENV),
        flush=True,
    )
    runs: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []
    for label, config, docs in planned:
        try:
            runs.append(_warm(label, config, args, docs))
        except Exception as error:  # a failed stage must not hide the warmed ones
            print(f"[prewarm] {label}: FAILED -- {type(error).__name__}: {error}", flush=True)
            failed.append((label, f"{type(error).__name__}: {error}"))
            torch.cuda.empty_cache()
            if args.fail_fast:
                break

    rows = probe.rows()
    print("\n[prewarm] first-launch cost per variant (largest first):")
    for row in rows:
        steady = f"{row['steady_ms']:.1f} ms/launch" if row["steady_ms"] else "not re-run"
        print(f"  {row['compile_seconds']:>8,.1f}s  {row['variant']:<44} {steady}")
    print(
        f"[prewarm] {len(rows)} variant(s), {sum(row['compile_seconds'] for row in rows):,.0f}s of compilation"
    )

    if failed:
        print("\n[prewarm] stages that did NOT warm (fix before relying on the cache):")
        for label, message in failed:
            print(f"  {label}: {message}")
        print("[prewarm] note: kernels warmed by the successful stages stay cached, rerun resumes")

    if args.json_out:
        payload = {
            "cache": dict(CACHE_ENV),
            "variants": rows,
            "runs": runs,
            "failed": [{"label": label, "error": message} for label, message in failed],
        }
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
