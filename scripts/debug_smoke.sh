#!/usr/bin/env bash
# End-to-end debug-model gate: smoke -> consistency -> tests.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=.venv/bin/python

echo "== 1/3 training smoke =="
$PY -m minideepseek.train.smoke \
  --flavor debugmodel --seq-len 128 --n-mtp-layers 1 --steps 2 \
  --device cpu --out-dir checkpoints/smoke

echo "== 2/3 consistency gate =="
$PY -m minideepseek.consistency.check --flavor debugmodel --seq-len 128 --demo

echo "== 3/3 test suite =="
$PY -m pytest -q
