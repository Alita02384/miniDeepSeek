#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/pretrain/full_6b.yaml}"
exec .venv/bin/python -m minideepseek.train.run --config "$CONFIG" --mode pretrain
