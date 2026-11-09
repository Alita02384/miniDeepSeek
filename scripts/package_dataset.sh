#!/usr/bin/env bash
# Package the flat JSONL data for upload (no tokenization artifacts).
#
# The 3B pretraining set is exactly three files:
#   pretrain_general.jsonl  MiniMind + external, exact-deduped and capped
#   pretrain_code.jsonl     deduped and capped
#   pretrain_tool.jsonl     deduped and capped
# ``pretrain_t2t.jsonl`` is merged into pretrain_general.jsonl and is excluded
# even if an old copy is still lying around.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${1:-dist/minideepseek-data}"

for name in pretrain_general pretrain_code pretrain_tool; do
  if [ ! -s "data/$name.jsonl" ]; then
    echo "missing or empty data/$name.jsonl" >&2
    echo "run scripts/dedup_and_split.py before packaging" >&2
    exit 1
  fi
done
if [ -f data/pretrain_t2t.jsonl ]; then
  echo "WARNING: data/pretrain_t2t.jsonl exists; it is merged into" >&2
  echo "         pretrain_general.jsonl and will NOT be packaged." >&2
fi

rm -rf "$OUT"
mkdir -p "$OUT/manifests"
cp -a data/*.jsonl "$OUT/"
rm -f "$OUT/pretrain_t2t.jsonl"
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
# miniDeepSeek 145M data

Flat JSONL dataset.  Tokenization is generated on the training host by
`scripts/build_pretrain_packs.sh` and is intentionally not packaged.

## Pretraining (3B tokens, exact-deduped and capped)

| file | tokens | purpose |
|---|---:|---|
| pretrain_general.jsonl | ~2.46B | MiniMind + fineweb-edu/chinese-fineweb-edu/cosmopedia/finemath |
| pretrain_code.jsonl | ~450M | codeparrot-clean (permissive, <=200 lines) |
| pretrain_tool.jsonl | ~90M | ToolACE + glaive schema text |

The three files are contiguous, disjoint token ranges for the 2K -> 4K -> 8K
curriculum (`scripts/build_pretrain_packs.sh`).  `pretrain_t2t.jsonl` is merged
into `pretrain_general.jsonl` and is not shipped.

## SFT / GRPO / OPD / eval

| file | purpose |
|---|---|
| sft_general.jsonl / sft_code.jsonl / sft_tool.jsonl | specialist SFT |
| grpo_general_train.jsonl / grpo_code_train.jsonl / grpo_tool_train.jsonl | GRPO prompt pools |
| eval_general.jsonl / eval_code.jsonl / eval_tool.jsonl | held-out eval (never train) |
| opd_prompts_general.jsonl / opd_prompts_tool.jsonl / opd_prompts_mixed.jsonl | OPD prompts |
| identity_sft.jsonl | canonical identity answers appended to SFT |

Provenance: MiniMind corpora are CC-BY-NC-4.0; external corpora carry their own
licenses (see doc/external_data.md).  `eval_*` is evaluation-only.
MD
echo "packaged $OUT ($(du -sh "$OUT" | cut -f1))"
