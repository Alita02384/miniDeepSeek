#!/usr/bin/env bash
# GRPO launcher. `torchrun` + manual DDP when NPROC_PER_NODE>1 (CUDA target).
# Cached/KV rollout keeps one full model per rank, so FSDP2 is not supported
# here: configs/grpo/*.yaml set distributed.mode=ddp.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MINIDEEPSEEK_ATTN_BACKEND="${MINIDEEPSEEK_ATTN_BACKEND:-torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CONFIG="${1:-configs/grpo/general.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [ "$NPROC_PER_NODE" -gt 1 ] || [ -n "${WORLD_SIZE:-}" ]; then
  # ``python -m`` instead of ``.venv/bin/torchrun``: the runbook's venv
  # reuses the image's system torch (``--system-site-packages``), which puts
  # the ``torchrun`` console script in the *system* bin dir, not in the venv.
  exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m minideepseek.rl.run grpo --config "$CONFIG"
fi
exec .venv/bin/python -m minideepseek.rl.run grpo --config "$CONFIG"
