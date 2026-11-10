#!/usr/bin/env bash
# Multi-stage 2K -> 4K -> 8K curriculum. Prepare each stage .npz first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/pretrain/length_curriculum.yaml}"

# Kernel caches live inside the repo (git-ignored).  A cold Triton cache costs
# 20-40 minutes of gather-attention compilation on step 0; `scripts/prewarm_attn.py`
# fills it once and every later start reuses it (see minideepseek.paths.CACHE_DIR).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.cache/inductor}"
export ATTN_GYM_CUTE_CACHE_DIR="${ATTN_GYM_CUTE_CACHE_DIR:-$ROOT/.cache/attn_gym}"
# Router diagnostics are read every micro-batch; keeping the recorded tensors on
# the GPU instead of copying them to the host removes 21 blocking D2H syncs per
# micro-batch (672 per optimizer step at grad_accum=32).  See decisions.md D72.
export MINIDEEPSEEK_ROUTING_GPU="${MINIDEEPSEEK_ROUTING_GPU:-1}"
# Attention runs fp32 unless the kernel casts at its boundary, and attn_gym's
# fp32 path does not use tensor cores: 443 ms -> 17 ms per micro-batch on a
# 4090 (2K stage, K=64).  The bf16 variants must be prewarmed first
# (`prewarm_attn.py`, one window per stage); 8K's K=256 is the slow one.
export MINIDEEPSEEK_ATTN_BF16="${MINIDEEPSEEK_ATTN_BF16:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [ "$NPROC_PER_NODE" -gt 1 ] || [ -n "${WORLD_SIZE:-}" ]; then
  # ``python -m`` instead of ``.venv/bin/torchrun``: the runbook's venv
  # reuses the image's system torch (``--system-site-packages``), which puts
  # the ``torchrun`` console script in the *system* bin dir, not in the venv.
  exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
    scripts/train_curriculum.py --config "$CONFIG"
fi
exec .venv/bin/python scripts/train_curriculum.py --config "$CONFIG"
