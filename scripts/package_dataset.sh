#!/usr/bin/env bash
# Package the flat JSONL data for upload (no tokenization artifacts).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-dist/minideepseek-data}"
rm -rf "$OUT"
mkdir -p "$OUT/manifests"
cp -a data/*.jsonl "$OUT/"
cp -a data/manifests/* "$OUT/manifests/" 2>/dev/null || true
(
  cd "$OUT"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum ./*.jsonl > SHA256SUMS
  else
    shasum -a 256 ./*.jsonl > SHA256SUMS
  fi
)
cat > "$OUT/README.md" <<'MD'
# miniDeepSeek specialist data

Flat JSONL dataset built from the MiniMind data checkout.

| file | purpose |
|---|---|
| pretrain_t2t.jsonl | pretraining |
| sft_general.jsonl | general SFT specialist |
| sft_tool.jsonl | tool-use SFT specialist (tool subset + general slice) |
| grpo_general_train.jsonl / eval_general.jsonl | general RLAIF prompts / eval |
| grpo_tool_train.jsonl / eval_tool.jsonl | multi-turn tool prompts / eval |
| opd_prompts_general.jsonl / opd_prompts_tool.jsonl / opd_prompts_mixed.jsonl | OPD prompt pool |

`tokenization/` is generated on the training host and is intentionally not
packaged.  Pretrain/SFT data is derived from MiniMind (`CC-BY-NC-4.0` for the
pretrain/SFT corpora; `agent_rl*`/`rlaif` provenance is mixed, check upstream).
MD
echo "packaged $OUT"
