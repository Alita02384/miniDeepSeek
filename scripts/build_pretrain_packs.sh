#!/usr/bin/env bash
# Build per-domain packed datasets for the 3B / 2K-4K-8K curriculum.
# Token budget: general 2.46B + code 0.45B + tool 0.087B = ~3.0B (disjoint per stage).
# The 8K ranges end ``VAL_TAIL`` tokens early; that tail is the held-out
# validation set (val_*_*_v2.npz), carved afterwards by scripts/make_val_split.py
# because the 1M-token head-of-corpus sets were not representative (D84).
# The three source files are already deduped and capped by scripts/dedup_and_split.py.
# Run this on the training host (packing tokenizes ~3B tokens; not needed locally).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=data/tokenization
mkdir -p "$OUT"
GEN="data/pretrain_general.jsonl"

pack () { # domain inputs seq_len skip cap out
  local domain=$1 inputs=$2 seq=$3 skip=$4 cap=$5 out=$6
  echo "=== $domain seq=$seq skip=$skip cap=$cap -> $out"
  $PY scripts/pack_domain.py --input $inputs --seq-len "$seq" \
      --skip-tokens "$skip" --max-tokens "$cap" --domain "$domain" --out "$OUT/$out"
}

# validation: 1M tokens from the head of each domain, per stage length.
# Superseded for the 4K/8K stages by val_*_*_v2.npz (see the tail carve below);
# kept because stage 0 already trained against the 2K sets.
for spec in "2k 2048" "4k 4096" "8k 8192"; do
  set -- $spec; tag=$1; seq=$2
  pack general "$GEN" "$seq" 0 1000000 "val_general_${tag}.npz"
  pack code    "data/pretrain_code.jsonl" "$seq" 0 1000000 "val_code_${tag}.npz"
  pack tool    "data/pretrain_tool.jsonl" "$seq" 0 1000000 "val_tool_${tag}.npz"
done

# train stages: disjoint token ranges.  The 8K ranges stop VAL_TAIL early so
# their last tokens can become the held-out validation tail.
VAL_TAIL_GENERAL=40000000
VAL_TAIL_CODE=16000000
VAL_TAIL_TOOL=6000000

pack general "$GEN" 2048   1000000   369000000  pretrain_general_2k.npz
pack general "$GEN" 4096   370000000 1353000000 pretrain_general_4k.npz
pack general "$GEN" 8192  1723000000 $((738000000 - VAL_TAIL_GENERAL)) pretrain_general_8k.npz

pack code "data/pretrain_code.jsonl" 2048   1000000   67500000 pretrain_code_2k.npz
pack code "data/pretrain_code.jsonl" 4096   68500000  247500000 pretrain_code_4k.npz
pack code "data/pretrain_code.jsonl" 8192  316000000 $((135000000 - VAL_TAIL_CODE)) pretrain_code_8k.npz

pack tool "data/pretrain_tool.jsonl" 2048   1000000   13000000 pretrain_tool_2k.npz
pack tool "data/pretrain_tool.jsonl" 4096   14000000  48000000 pretrain_tool_4k.npz
pack tool "data/pretrain_tool.jsonl" 8192   62000000  26000000 pretrain_tool_8k.npz

# Held-out validation tails, packed per stage length.  They are disjoint from
# every training range above because the 8K pack stopped VAL_TAIL early.
for spec in "4k 4096" "8k 8192"; do
  set -- $spec; tag=$1; seq=$2
  $PY scripts/make_val_split.py --source "$OUT/pretrain_general_8k.npz" \
      --tokens "$VAL_TAIL_GENERAL" --seq-len "$seq" --domain general \
      --out "$OUT/val_general_${tag}_v2.npz"
  $PY scripts/make_val_split.py --source "$OUT/pretrain_code_8k.npz" \
      --tokens "$VAL_TAIL_CODE" --seq-len "$seq" --domain code \
      --out "$OUT/val_code_${tag}_v2.npz"
  $PY scripts/make_val_split.py --source "$OUT/pretrain_tool_8k.npz" \
      --tokens "$VAL_TAIL_TOOL" --seq-len "$seq" --domain tool \
      --out "$OUT/val_tool_${tag}_v2.npz"
done

echo "=== done"
du -sh "$OUT"
