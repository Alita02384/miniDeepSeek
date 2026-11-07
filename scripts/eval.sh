#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/eval/general.yaml}"
exec .venv/bin/python -m minideepseek.eval.run --config "$CONFIG"
