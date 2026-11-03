#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/opd/three_teacher.yaml}"
exec .venv/bin/python -m minideepseek.rl.run opd --config "$CONFIG"
