#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/sft/math.yaml}"
exec .venv/bin/python -m minideepseek.train.run --config "$CONFIG" --mode sft
