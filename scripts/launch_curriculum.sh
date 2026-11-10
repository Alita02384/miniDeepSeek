#!/usr/bin/env bash
# Launch a curriculum run: one log file and one metrics directory per launch.
#
#   bash scripts/launch_curriculum.sh [config] [--gpus N]
#
# Everything after the banner goes to logs/<run_id>_<config>.log instead of a
# fixed logs/curriculum_145m.log, so a restart never truncates or interleaves
# with the previous attempt's log.  The same run id is exported as
# MINIDEEPSEEK_RUN_ID, so the trainer puts its metrics in
# <stage>/runs/<run_id>_<flavor>_<params>_<gpus>g/ rather than appending to a
# shared metrics.jsonl.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/pretrain/generated/curriculum_145m_g1.yaml"
GPUS="${NPROC_PER_NODE:-}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="${2:?--gpus needs a count}"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) CONFIG="$1"; shift ;;
  esac
done
if [ ! -f "$CONFIG" ]; then
  echo "no such config: $CONFIG" >&2
  exit 2
fi
if [ -z "$GPUS" ]; then
  GPUS="$(.venv/bin/python -c "import torch; print(max(1, torch.cuda.device_count()))" 2>/dev/null || echo 1)"
fi

mkdir -p logs
# ``logs/<config>_<stamp>.log``: the stamp is the *campaign* stamp kept in
# ``logs/.<config>.stamp``, so an interrupted training appends to the log it
# already had instead of spawning one file per restart.  A different config
# (different model) or a run with no checkpoint left gets a new stamp.
LOG="$(.venv/bin/python -m minideepseek.runs logpath "$CONFIG" --logs-dir logs)"
STAMP="${LOG%.log}"
STAMP="${STAMP##*_}"
APPENDING=no
[ -s "$LOG" ] && APPENDING=yes

printf 'run id  : %s\nconfig  : %s\ngpus    : %s\nlog     : %s\nappending: %s\n\n' \
  "$STAMP" "$CONFIG" "$GPUS" "$LOG" "$APPENDING"

export MINIDEEPSEEK_RUN_ID="$STAMP"
export NPROC_PER_NODE="$GPUS"
# From here on stdout/stderr go to the campaign log; the banner above stays on
# the terminal (and in nohup.out if the caller redirects).
exec >>"$LOG" 2>&1
echo "== start $(date -u +%FT%TZ)  config=$CONFIG  gpus=$GPUS  run_id=$STAMP  appending=$APPENDING"
if [ "$GPUS" -gt 1 ] || [ -n "${WORLD_SIZE:-}" ]; then
  exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
    scripts/train_curriculum.py --config "$CONFIG"
fi
exec .venv/bin/python scripts/train_curriculum.py --config "$CONFIG"
