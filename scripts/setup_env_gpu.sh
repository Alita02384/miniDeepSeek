#!/usr/bin/env bash
# Bootstrap a fresh gpuhome instance: venv on the system disk, project deps,
# CUDA sanity check.  Idempotent -- safe to re-run after recreating the box.
#
#   bash scripts/setup_env_gpu.sh          # env + GPU check
#   bash scripts/setup_env_gpu.sh --test   # ... and the CPU pytest suite
#
# Why this exists: every new instance gets a fresh system disk, so
# /opt/mds-venv and ~/.ssh/authorized_keys disappear with it.  Keeping the
# whole install in one script turns "换实例" into one command.
#
# The image already ships a CUDA torch (2.12.1+cu130 on the 2026-11 gpuhome
# image); we deliberately reuse it through --system-site-packages instead of
# downloading ~2.5 GB from download.pytorch.org.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-/opt/mds-venv}"
PY="${PY:-/opt/conda/bin/python}"
INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

[ -x "$PY" ] || { echo "ERROR: no base python at $PY (set PY=...)" >&2; exit 1; }

if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating $VENV from $PY (--system-site-packages)"
  "$PY" -m venv --system-site-packages "$VENV"
fi

if [ -L .venv ]; then
  ln -sfn "$VENV" .venv
elif [ -e .venv ]; then
  echo "ERROR: .venv exists and is not a symlink; remove it first" >&2
  exit 1
else
  ln -s "$VENV" .venv
fi

# Refuse to let pip pull a 2.5 GB torch because the image torch is invisible.
if ! "$VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: $VENV cannot import torch from $PY." >&2
  echo "       Fix the base image, or install torch explicitly; not doing it here." >&2
  exit 1
fi

echo "== installing project deps"
PIP_INDEX_URL="$INDEX" PIP_NO_CACHE_DIR=1 "$VENV/bin/pip" install -U pip
PIP_INDEX_URL="$INDEX" PIP_NO_CACHE_DIR=1 "$VENV/bin/pip" install -e '.[dev,hf]'

# With --system-site-packages pip treats the image's protobuf as already
# satisfying `protobuf>=4`, so it never installs one; sentencepiece then ships a
# newer gencode than the runtime it finds at import time and the reward model's
# tokenizer dies with "gencode 7.36.1 / runtime 7.35.1".  Force a venv-local
# protobuf so the venv copy shadows site-packages.
echo "== pinning a venv-local protobuf"
PIP_INDEX_URL="$INDEX" PIP_NO_CACHE_DIR=1 "$VENV/bin/pip" install --ignore-installed 'protobuf>=5'

echo "== hf check"
"$VENV/bin/python" - <<'PY'
import einops
import google.protobuf
import sentencepiece
import transformers

print(
    "transformers", transformers.__version__,
    "| sentencepiece", sentencepiece.__version__,
    "| protobuf", google.protobuf.__version__,
    "| einops", einops.__version__,
)
PY

echo "== torch check"
"$VENV/bin/python" - <<'PY'
import torch

print(
    "torch", torch.__version__,
    "| cuda", torch.version.cuda,
    "| available", torch.cuda.is_available(),
    "| devices", torch.cuda.device_count(),
)
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(
            "  gpu", index, torch.cuda.get_device_name(index),
            "| bf16", torch.cuda.is_bf16_supported(),
            "|", round(torch.cuda.get_device_properties(index).total_memory / 2**30, 1), "GiB",
        )
else:
    print("WARNING: torch.cuda.is_available() is False.")
    print("  'CUDA driver version is insufficient' means the driver is older than the")
    print("  wheel's CUDA build.  Install a matching wheel, e.g.:")
    print("  .venv/bin/pip install torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128")
PY

du -sh "$VENV" | sed 's/^/== /'

if [ "${1:-}" = "--test" ]; then
  echo "== pytest"
  "$VENV/bin/python" -m pytest -q
fi

echo
echo "next: rsync the data packs (doc/run_145m_gpuhome.md §5), then"
echo "      .venv/bin/python scripts/autoconfig_145m.py --gpus N"
