#!/usr/bin/env bash
# Push a miniDeepSeek training box to a fresh instance.
#
# Run this **on the old box**.  It copies the repo (code/configs/scripts), the
# tokenized packs + JSONL, the checkpoints and the Triton kernel cache to the
# new host, then prints the commands to run there.
#
#   bash scripts/migrate_box.sh <new-host> <new-port> [--data] [--no-cache]
#
# --data      also copy data/tokenization (packs + val, ~26 GB) and data/*.jsonl
# --no-cache  skip ~/.triton/cache (it only helps when the new box has the same
#             GPU architecture and the same Triton version as this one)
#
# Resume on the new box:
#   bash scripts/setup_env_gpu.sh
#   .venv/bin/python scripts/autoconfig_145m.py --gpus N
#   NPROC_PER_NODE=N bash scripts/train_curriculum.sh configs/pretrain/generated/curriculum_145m_gN.yaml
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <new-host> <new-port> [--data] [--no-cache]" >&2
  exit 2
fi
NEW_HOST="$1"
NEW_PORT="$2"
shift 2

WITH_DATA=0
WITH_CACHE=1
for arg in "$@"; do
  case "$arg" in
    --data) WITH_DATA=1 ;;
    --no-cache) WITH_CACHE=0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SSH_CMD="ssh -p ${NEW_PORT}"
DEST="root@${NEW_HOST}:/root/rivermind-data/miniDeepSeek"

echo "== 1/4 code -> ${DEST}"
rsync -avz --partial --progress -e "$SSH_CMD" \
  --exclude '.venv' --exclude '.cache' --exclude '.pytest_cache' --exclude '.ruff_cache' \
  --exclude 'data' --exclude 'download' --exclude 'dist' --exclude 'checkpoints' \
  --exclude 'runs' --exclude 'reports' --exclude 'internlm2-1_8b-reward' \
  --exclude '__pycache__' --exclude '*.tar.zst' \
  ./ "${DEST}/"

if [ "$WITH_DATA" = 1 ]; then
  echo "== 2/4 data (tokenized packs + jsonl)"
  if [ -d data/tokenization ]; then
    rsync -avz --partial --progress -e "$SSH_CMD" \
      data/tokenization/ "${DEST}/data/tokenization/"
  fi
  shopt -s nullglob
  jsonl=(data/*.jsonl)
  shopt -u nullglob
  if [ "${#jsonl[@]}" -gt 0 ]; then
    rsync -avz --partial --progress -e "$SSH_CMD" "${jsonl[@]}" "${DEST}/data/"
  fi
else
  echo "== 2/4 data skipped (pass --data to copy packs)"
fi

echo "== 3/4 checkpoints"
if [ -d checkpoints ]; then
  rsync -avz --partial --progress -e "$SSH_CMD" checkpoints/ "${DEST}/checkpoints/"
else
  echo "   no checkpoints/ yet"
fi

if [ "$WITH_CACHE" = 1 ] && [ -d "$ROOT/.cache" ]; then
  echo "== 4/4 kernel cache ($(du -sh "$ROOT/.cache" | cut -f1))"
  echo "   only reused when GPU arch and torch/Triton versions match; a miss just recompiles"
  rsync -a --partial --progress -e "$SSH_CMD" \
    "$ROOT/.cache/" "root@${NEW_HOST}:/root/rivermind-data/miniDeepSeek/.cache/"
else
  echo "== 4/4 kernel cache skipped"
fi

cat <<EOF

done. On the new box:

  cd /root/rivermind-data/miniDeepSeek
  bash scripts/setup_env_gpu.sh
  GPUS=\$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count())')
  .venv/bin/python scripts/autoconfig_145m.py --gpus "\$GPUS"
  export NCCL_P2P_DISABLE=1 MINIDEEPSEEK_ATTN_BACKEND=auto
  # Compile once, visibly, into .cache/ ; then every start is instant.
  .venv/bin/python scripts/prewarm_attn.py \
    --curriculum "configs/pretrain/generated/curriculum_145m_g\${GPUS}.yaml"
  NPROC_PER_NODE="\$GPUS" bash scripts/train_curriculum.sh \\
    "configs/pretrain/generated/curriculum_145m_g\${GPUS}.yaml"
EOF
