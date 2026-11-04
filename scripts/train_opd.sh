#!/usr/bin/env bash
# Multi-teacher OPD launcher. `torchrun` + FSDP2 when NPROC_PER_NODE>1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/opd/three_teacher.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [ "$NPROC_PER_NODE" -gt 1 ] || [ -n "${WORLD_SIZE:-}" ]; then
  exec .venv/bin/torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m minideepseek.rl.run opd --config "$CONFIG"
fi
exec .venv/bin/python -m minideepseek.rl.run opd --config "$CONFIG"
