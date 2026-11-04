#!/usr/bin/env bash
# Create the local venv and install the runtime.
#
# The project is self-contained: model code lives under src/minideepseek/model
# and no sibling checkout is imported at runtime.  CUDA is optional here; the
# same environment works on CPU for the debug model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3.12}"
UV="${UV:-$HOME/.local/bin/uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"

if ! command -v "$UV" >/dev/null 2>&1; then
  echo "uv not found at $UV; install it or set UV=/path/to/uv" >&2
  exit 1
fi

"$UV" venv --python "$PYTHON" .venv
"$UV" pip install --python .venv/bin/python \
  torch safetensors tokenizers jinja2 numpy pyyaml pytest ruff
"$UV" pip install --python .venv/bin/python -e . --no-deps

.venv/bin/python - <<'PY'
import torch
from minideepseek.model import build_model
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
model = build_model("debugmodel", seq_len=128, n_mtp_layers=1, device="cpu")
print("debugmodel params:", sum(p.numel() for p in model.parameters()))
PY

echo "Environment ready. Run: ./scripts/debug_smoke.sh"
