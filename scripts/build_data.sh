#!/usr/bin/env bash
# Build the flat JSONL dataset from a local MiniMind checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -m minideepseek.data.build \
  --minimind-root "${MINIMIND_ROOT:-../minimind/dataset}" \
  --out "${DATA_DIR:-data}" "$@"
