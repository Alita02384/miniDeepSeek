#!/usr/bin/env python3
"""Compare the self-contained V4 model against the torchtitan reference.

This is a **development-only** script: it needs a local ``../torchtitan``
checkout and is not part of the installed package or the default test suite.
It builds the same debug config on both sides, copies the reference state dict
into our model, feeds identical inputs, and reports per-layer relative error.

The expected residual is ~1e-4 in float32 because the reference casts routed
expert inputs to bfloat16 even in a float32 model (its grouped GEMM requires
bf16).  Run with ``--tolerance`` to change the acceptance threshold.

Usage:
    .venv/bin/python scripts/verify_reference.py
    .venv/bin/python scripts/verify_reference.py --seq-len 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
TORCHTITAN = WORKSPACE / "torchtitan"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument("--tolerance", type=float, default=2e-3)
    parser.add_argument("--mtp", action="store_true", help="also compare MTP logits")
    args = parser.parse_args()

    if not (TORCHTITAN / "torchtitan" / "models" / "deepseek_v4").is_dir():
        print(f"reference checkout not found at {TORCHTITAN}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(TORCHTITAN))

    import torch
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.deepseek_v4.flavors import build_model_config as ref_config
    from torchtitan.models.deepseek_v4.model import DeepSeekV4Model as RefModel

    from minideepseek.model.config import build_model_config
    from minideepseek.model.model import DeepSeekV4Model

    torch.manual_seed(0)
    seq = args.seq_len
    ref_cfg = ref_config(
        "debugmodel",
        parallelism=ParallelismConfig(enable_sequence_parallel=False),
        seq_len=seq,
        n_mtp_layers=1 if args.mtp else 0,
    )
    reference = RefModel(ref_cfg)
    reference.init_states(buffer_device=torch.device("cpu"))
    reference.eval()

    config = build_model_config("debugmodel", seq_len=seq, n_mtp_layers=1 if args.mtp else 0)
    model = DeepSeekV4Model(config)
    # D34 removed the dead load-balance bias from hash layers; the reference
    # checkpoint may still carry those keys.  Keep the intersection with this
    # model's expected keys (all real mismatches still fail strict load).
    our_keys = set(model.state_dict())
    model.load_state_dict(
        {key: value for key, value in reference.state_dict().items() if key in our_keys},
        strict=True,
    )
    model.eval()

    reference_params = sum(p.numel() for p in reference.parameters())
    our_params = sum(p.numel() for p in model.parameters())
    if reference_params != our_params:
        print(f"parameter count mismatch: {reference_params} vs {our_params}", file=sys.stderr)
        return 1

    # Two packed documents exercise the cross-document attention boundaries.
    lengths = [seq // 2, seq - seq // 2]
    positions = torch.cat([torch.arange(length) for length in lengths])
    tokens = torch.randint(0, 2048, (seq,), generator=torch.Generator().manual_seed(3))
    metadata = reference._get_attention_metadata(positions)

    worst_block = 0.0
    worst_logit_mean = 0.0
    worst_logit_max = 0.0
    with torch.no_grad():
        hidden = reference.tok_embeddings(tokens).unsqueeze(1).repeat(1, reference.hc_mult, 1)
        for layer_id in range(len(config.layers)):
            ref_layer = reference.layers[str(layer_id)]
            our_layer = model.layers[str(layer_id)]
            inner_type = type(ref_layer.attention.inner_attention)
            ref_out = ref_layer(hidden, tokens, metadata[inner_type], positions)
            our_out = our_layer(hidden, tokens, model._get_attention_metadata(positions), positions)
            rel = float((ref_out - our_out).abs().max() / ref_out.abs().max().clamp_min(1e-12))
            worst_block = max(worst_block, rel)
            print(f"layer {layer_id}: max rel diff {rel:.3e}")
            hidden = ref_out

        if args.mtp:
            ref_logits = reference(tokens, positions, metadata)
            our_logits = model(tokens, positions, model._get_attention_metadata(positions))
            for depth, (ref, ours) in enumerate(zip(ref_logits, our_logits, strict=True)):
                diff = (ref - ours).abs()
                mean_abs = float(diff.mean())
                max_abs = float(diff.max())
                worst_logit_mean = max(worst_logit_mean, mean_abs)
                worst_logit_max = max(worst_logit_max, max_abs)
                print(
                    f"logits depth {depth}: mean abs {mean_abs:.3e}, "
                    f"max abs {max_abs:.3e}"
                )

    print(
        f"worst block relative diff: {worst_block:.3e} "
        f"(tolerance {args.tolerance:.0e})"
    )
    if args.mtp:
        print(
            f"logits: worst mean abs {worst_logit_mean:.3e}, "
            f"worst max abs {worst_logit_max:.3e}"
        )
    ok = worst_block <= args.tolerance
    if args.mtp:
        ok = ok and worst_logit_mean <= args.tolerance
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
