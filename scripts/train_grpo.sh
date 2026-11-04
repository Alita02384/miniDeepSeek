#!/usr/bin/env bash
# GRPO launcher (single process torch rollout). Multi-GPU RL is expected to go
# through the verl adapter; see doc/decisions.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/grpo/math.yaml}"
exec .venv/bin/python -m minideepseek.rl.run grpo --config "$CONFIG"
