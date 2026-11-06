#!/usr/bin/env python3
"""Measure activation memory and FLOPs per token for a model configuration.

The number that decides whether a 24 GB card can train a config is *activation*
memory, not parameter memory.  This script runs one real ``compute_loss`` step
(same path as training: MTP + indexer losses included) and accounts for every
tensor saved for backward, counting each storage once.

Reported values:

* ``saved_mib`` — unique storages kept alive until backward;
* ``mib_per_token`` — the linear coefficient used to extrapolate to longer
  sequence lengths (attention gathers grow with tokens x top-k, so treat the
  extrapolation as an estimate, not a measurement);
* ``fwd_mflops_per_token`` — matmul FLOPs of the forward pass only.

Examples::

    # CPU proxy for a CUDA bf16 run (autocast is forced on both devices)
    .venv/bin/python scripts/measure_activation_budget.py --seq-len 1024

    # On the target GPU: real allocator numbers, run this before a long job
    .venv/bin/python scripts/measure_activation_budget.py --device cuda \\
        --seq-len 4096 --flavor mini_220m_a80m --json-out /tmp/budget.json

    # All three shipped architectures at the training sequence length
    .venv/bin/python scripts/measure_activation_budget.py --seq-len 4096
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minideepseek.config import ModelSettings, load_config, resolve_device  # noqa: E402
from minideepseek.model import build_model, from_sequences  # noqa: E402

FLAVORS = ("mini_60m_dense_d768", "mini_145m_a60m", "mini_220m_a80m")


class _SavedTensorCounter:
    """Attribute saved-for-backward bytes to the enclosing top-level module."""

    def __init__(self) -> None:
        self.stack: list[str] = ["<root>"]
        self.storages: dict[tuple[int, int], dict] = {}
        self.by_bucket: dict[str, int] = {}

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset())
        if key not in self.storages:
            nbytes = tensor.numel() * tensor.element_size()
            module = self.stack[-1]
            bucket = ".".join(module.split(".")[:2]) or "<root>"
            self.storages[key] = {
                "bytes": nbytes,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "module": module,
            }
            self.by_bucket[bucket] = self.by_bucket.get(bucket, 0) + nbytes
        return tensor

    @staticmethod
    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @property
    def total(self) -> int:
        return sum(entry["bytes"] for entry in self.storages.values())


def _attach_hooks(model: torch.nn.Module, counter: _SavedTensorCounter) -> list:
    handles = []

    def push_scope(name: str):
        def hook(_module, _args):
            counter.stack.append(name)

        return hook

    def pop_scope(name: str):
        def hook(_module, _args, _output):
            while counter.stack and counter.stack[-1] != name:
                counter.stack.pop()
            if counter.stack:
                counter.stack.pop()

        return hook

    for name, module in model.model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_pre_hook(push_scope(name)))
        handles.append(module.register_forward_hook(pop_scope(name)))
    return handles


def measure(
    flavor: str,
    seq_len: int,
    *,
    device: torch.device,
    dtype: str,
    batch_docs: int = 1,
) -> dict:
    settings = load_config(ModelSettings, ROOT / "configs" / "model" / f"{flavor}.yaml")
    settings.device = str(device)
    settings.seq_len = seq_len
    settings.max_context = max(settings.max_context, seq_len)
    # Derive index_topk from the sequence length, exactly like a training run.
    settings.index_topk = None

    model = build_model(settings, device=device, dtype=None)
    generator = torch.Generator().manual_seed(0)
    sequences = [
        torch.randint(
            3, max(int(settings.vocab_size) - 1, 8), (seq_len,), generator=generator
        ).tolist()
        for _ in range(max(batch_docs, 1))
    ]
    batch = from_sequences(sequences, device=device)

    counter = _SavedTensorCounter()
    handles = _attach_hooks(model, counter)
    autocast_dtype = torch.bfloat16 if dtype == "bfloat16" else None
    try:
        with (
            torch.enable_grad(),
            torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
            ),
            torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack),
            FlopCounterMode(display=False) as flops,
        ):
            output = model.compute_loss(batch, mtp_weight=0.3)
        forward_flops = flops.get_total_flops()
    finally:
        for handle in handles:
            handle.remove()

    tokens = seq_len * max(batch_docs, 1)
    report = model.parameter_report()
    result = {
        "flavor": flavor,
        "seq_len": seq_len,
        "docs": batch_docs,
        "tokens": tokens,
        "device": str(device),
        "dtype": dtype,
        "params_total": int(report["total"]),
        "params_active": int(report["active_estimate"]),
        "saved_mib": counter.total / 2**20,
        "mib_per_token": counter.total / tokens / 2**20,
        "fwd_mflops_per_token": forward_flops / tokens / 1e6,
        "saved_by_bucket_mib": {k: v / 2**20 for k, v in sorted(counter.by_bucket.items())},
        "loss": output.total.detach().item(),
    }
    if device.type == "cuda":
        result["cuda_peak_alloc_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        result["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flavor",
        action="append",
        choices=FLAVORS,
        help="model YAML stem (repeatable; default: all three)",
    )
    parser.add_argument(
        "--seq-len",
        action="append",
        type=int,
        help="sequence length to measure (repeatable; default: 1024)",
    )
    parser.add_argument("--docs", type=int, default=1, help="documents per micro-batch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    flavors = args.flavor or list(FLAVORS)
    seq_lens = args.seq_len or [1024]
    results = []
    for flavor in flavors:
        for seq_len in seq_lens:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            result = measure(
                flavor,
                seq_len,
                device=device,
                dtype=args.dtype,
                batch_docs=args.docs,
            )
            results.append(result)
            peak = result.get("cuda_peak_alloc_mib")
            extra = f" cuda_peak={peak / 1024:.2f}GiB" if peak else ""
            print(
                f"{flavor:24s} seq={seq_len:5d} docs={args.docs} "
                f"saved={result['saved_mib'] / 1024:6.2f}GiB "
                f"({result['mib_per_token']:.3f} MiB/tok) "
                f"fwd={result['fwd_mflops_per_token']:.1f} MFLOP/tok{extra}",
                flush=True,
            )
            del result

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
