#!/usr/bin/env bash
# Server bootstrap: CUDA torch + project deps + data + CPU test.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/pip install -U pip
.venv/bin/pip install torch --index-url "$TORCH_INDEX_URL"
.venv/bin/pip install -e . --no-deps
.venv/bin/pip install safetensors tokenizers jinja2 numpy pyyaml pytest ruff
.venv/bin/pip install -e '.[hf]'

bash scripts/download_dataset.sh

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the server pipeline")
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

if [ -f data/SHA256SUMS ]; then
  (cd data && sha256sum -c SHA256SUMS)
fi
.venv/bin/python -m pytest -q
echo "Server environment ready. Next: bash scripts/run_pipeline.sh"
