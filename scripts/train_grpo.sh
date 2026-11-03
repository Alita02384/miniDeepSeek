#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/grpo/math.yaml}"
exec .venv/bin/python -m minideepseek.rl.run grpo --config "$CONFIG"
