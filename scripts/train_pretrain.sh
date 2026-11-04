#!/usr/bin/env bash
# Pretraining launcher. Set NPROC_PER_NODE>1 (CUDA) to use torchrun + FSDP2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/pretrain/full_6b.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [ "$NPROC_PER_NODE" -gt 1 ] || [ -n "${WORLD_SIZE:-}" ]; then
  exec .venv/bin/torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m minideepseek.train.run --config "$CONFIG" --mode pretrain
fi
exec .venv/bin/python -m minideepseek.train.run --config "$CONFIG" --mode pretrain
