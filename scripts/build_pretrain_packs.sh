#!/usr/bin/env bash
# Build per-domain packed datasets for the 3B / 2K-4K-8K curriculum.
# Token budget: general 2.46B + code 0.45B + tool 0.087B = ~3.0B (disjoint per stage).
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

# validation: 1M tokens from the head of each domain, per stage length
for spec in "2k 2048" "4k 4096" "8k 8192"; do
  set -- $spec; tag=$1; seq=$2
  pack general "$GEN" "$seq" 0 1000000 "val_general_${tag}.npz"
  pack code    "data/pretrain_code.jsonl" "$seq" 0 1000000 "val_code_${tag}.npz"
  pack tool    "data/pretrain_tool.jsonl" "$seq" 0 1000000 "val_tool_${tag}.npz"
done

# train stages: disjoint token ranges
pack general "$GEN" 2048   1000000   369000000  pretrain_general_2k.npz
pack general "$GEN" 4096   370000000 1353000000 pretrain_general_4k.npz
pack general "$GEN" 8192  1723000000  738000000 pretrain_general_8k.npz

pack code "data/pretrain_code.jsonl" 2048   1000000   67500000 pretrain_code_2k.npz
pack code "data/pretrain_code.jsonl" 4096   68500000  247500000 pretrain_code_4k.npz
pack code "data/pretrain_code.jsonl" 8192  316000000  135000000 pretrain_code_8k.npz

pack tool "data/pretrain_tool.jsonl" 2048   1000000   13000000 pretrain_tool_2k.npz
pack tool "data/pretrain_tool.jsonl" 4096   14000000  48000000 pretrain_tool_4k.npz
pack tool "data/pretrain_tool.jsonl" 8192   62000000  26000000 pretrain_tool_8k.npz

echo "=== done"
du -sh "$OUT"
