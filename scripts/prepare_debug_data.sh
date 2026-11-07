#!/usr/bin/env bash
# Build the small debug datasets from the minimind checkout.
# These are *samples*, not the 6B-token corpus; data expansion is out of scope
# for the current code-verification phase.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=.venv/bin/python
MINIMIND=../minimind

mkdir -p data/tokenization

$PY -m minideepseek.data.prepare pretrain \
  --input "$MINIMIND/dataset/pretrain_t2t_mini.jsonl" \
  --output data/tokenization/debug_pretrain.npz \
  --seq-len 128 --max-records 200 --min-chars 16 --vocab-limit 2048

$PY -m minideepseek.data.prepare sft \
  --input "$MINIMIND/dataset/sft_t2t_mini.jsonl" \
  --output data/tokenization/debug_sft.npz \
  --seq-len 256 --max-records 50 --vocab-limit 2048

$PY -m minideepseek.data.prepare grpo \
  --input "$MINIMIND/dataset/agent_rl_math.jsonl" \
  --output data/grpo_debug_math.jsonl \
  --domain math --verifier math_exact_match --max-records 20 --vocab-limit 2048

$PY - <<'PY'
import json
from pathlib import Path
from minideepseek.data.grpo import load_records

records = load_records("data/grpo_debug_math.jsonl")
opd = []
eval_rows = []
for index, record in enumerate(records):
    key = "math" if index % 2 == 0 else "code"
    opd.append(
        {
            "prompt": record.prompt,
            "teacher_key": key,
            "domain": "math" if key == "math" else "code",
            "source": "debug",
        }
    )
    eval_rows.append(
        {
            "prompt": record.prompt,
            "reference_answer": record.reference_answer,
            "domain": record.domain,
            "verifier": record.verifier,
        }
    )
path = Path("data/opd_debug_prompts.jsonl")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in opd) + "\n")
path = Path("data/eval_debug_math.jsonl")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in eval_rows[:10]) + "\n")
print("wrote", len(opd), "opd prompts and", len(eval_rows[:10]), "eval rows")
PY

echo "Debug data ready under data/."
