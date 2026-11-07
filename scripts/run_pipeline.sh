#!/usr/bin/env bash
# One-command training pipeline.
# Usage: NPROC_PER_NODE=4 bash scripts/run_pipeline.sh [pipeline.yaml] [--dry-run]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/pipeline/minimind_145m.yaml}"
shift || true
exec .venv/bin/python -m minideepseek.pipeline --config "$CONFIG" "$@"
