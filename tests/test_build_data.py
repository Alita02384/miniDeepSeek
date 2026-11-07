from __future__ import annotations

import json
from pathlib import Path

from minideepseek.data.build import build_all


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def test_build_flat_datasets(tmp_path):
    minimind = tmp_path / "minimind"
    minimind.mkdir()
    tools = [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}}]
    _write(
        minimind / "sft_t2t_mini.jsonl",
        [
            {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
            {
                "conversations": [
                    {"role": "system", "content": "", "tools": tools},
                    {"role": "user", "content": "1+1"},
                    {"role": "assistant", "content": "", "tool_calls": [{"name": "calc", "arguments": {}}]},
                    {"role": "tool", "content": "2"},
                    {"role": "assistant", "content": "2"},
                ]
            },
        ],
    )
    _write(
        minimind / "rlaif.jsonl",
        [
            {
                "conversations": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": ""},
                ]
            }
        ],
    )
    tool_record = {
        "conversations": [
            {"role": "system", "content": "", "tools": tools},
            {"role": "user", "content": "1+1"},
            {"role": "assistant", "content": ""},
        ],
        "gt": ["2"],
    }
    _write(minimind / "agent_rl.jsonl", [tool_record])
    _write(minimind / "agent_rl_math.jsonl", [tool_record])
    out = tmp_path / "data"
    report = build_all(minimind, out, skip_pretrain=True, max_per_domain=10)
    assert (out / "sft_general.jsonl").exists()
    assert (out / "sft_tool.jsonl").exists()
    assert (out / "grpo_general_train.jsonl").exists()
    assert (out / "grpo_tool_train.jsonl").exists()
    assert (out / "opd_prompts_mixed.jsonl").exists()
    assert report["sft_tool"]["tool_records"] == 1
    assert report["grpo_tool"]["records"] == 1
    general = [json.loads(line) for line in (out / "grpo_general_train.jsonl").read_text().splitlines()]
    assert general and general[0]["multi_turn"] is True
