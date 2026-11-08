"""Drop MiniMind identity records and install this project's identity.

The MiniMind corpora teach the model "我是由 jingyaogong 开发的高效小参数 AI 模型".
That phrasing leaks into SFT, GRPO prompts, OPD prompts and eval sets.  This
script removes every record that mentions the old identity and appends a small
canonical identity set for **mini-deepseek / chengjionghao**.

Usage::

    python scripts/scrub_identity.py            # rewrite data/*.jsonl in place
    python scripts/scrub_identity.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

#: Text markers of the old identity.  ``minimind`` also appears in benign prose,
#: but every hit we sampled was the identity block, so dropping the record is
#: safer than keeping a model that introduces itself as MiniMind.
PATTERNS = [
    re.compile(r"jingyaogong", re.I),
    re.compile(r"gongjy", re.I),
    re.compile(r"minimind", re.I),
    re.compile(r"小参数\s*AI\s*模型", re.I),
]

IDENTITY_FILE = DATA / "identity_sft.jsonl"
APPEND_TO = DATA / "sft_general.jsonl"

IDENTITY_PAIRS: list[tuple[str, str]] = [
    ("你是谁？", "我是 mini-deepseek，一个基于 DeepSeek-V4 架构的小型 MoE 语言模型，由 chengjionghao 训练。"),
    ("你叫什么名字？", "我叫 mini-deepseek。"),
    ("你是什么模型？", "我是 mini-deepseek，一个约 1.45 亿参数、每 token 激活约 6100 万参数的 MoE 语言模型，由 chengjionghao 训练。"),
    ("你是谁开发的？", "我是由 chengjionghao 训练的 mini-deepseek。"),
    ("谁训练了你？", "chengjionghao 训练了我，我的名字是 mini-deepseek。"),
    ("你的开发者是谁？", "chengjionghao。我是 mini-deepseek。"),
    ("你是哪个团队做的？", "我是 mini-deepseek，由 chengjionghao 独立训练，不属于任何公司。"),
    ("你是 DeepSeek 官方模型吗？", "不是。我是 mini-deepseek，采用 DeepSeek-V4 的公开架构与训练方法做的小规模复现，由 chengjionghao 训练，与 DeepSeek 官方无关。"),
    ("你和 DeepSeek 是什么关系？", "我参考 DeepSeek-V4 的架构与训练流程，但我是独立训练的 mini-deepseek，训练者是 chengjionghao。"),
    ("你是 GPT 吗？", "不是，我是 mini-deepseek，一个开源的 MoE 研究模型，由 chengjionghao 训练。"),
    ("你能做什么？", "我是 mini-deepseek，可以做中文和英文对话、回答常识问题、写和解释代码、以及按给定函数定义发起工具调用。"),
    ("你的训练数据是什么？", "我的训练数据包括中英文通用语料、代码和工具调用数据，由 chengjionghao 清洗和配比。"),
    ("你的参数有多大？", "mini-deepseek 总参数约 1.45 亿，MoE 每 token 激活约 6100 万，由 chengjionghao 训练。"),
    ("Who are you?", "I'm mini-deepseek, a small MoE language model built on the DeepSeek-V4 architecture and trained by chengjionghao."),
    ("Who trained you?", "chengjionghao trained me. My name is mini-deepseek."),
    ("What model are you?", "I'm mini-deepseek, a ~145M-parameter MoE model trained by chengjionghao."),
    ("Are you an official DeepSeek model?", "No. I'm mini-deepseek, an independent small-scale reproduction of the DeepSeek-V4 architecture, trained by chengjionghao."),
]


def is_identity(payload: str) -> bool:
    return any(pattern.search(payload) for pattern in PATTERNS)


def scrub(path: Path, *, dry_run: bool) -> dict[str, int]:
    tmp = path.with_suffix(path.suffix + ".tmp")
    kept = dropped = 0
    with path.open(encoding="utf-8") as source, tmp.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            if is_identity(line):
                dropped += 1
                continue
            if not dry_run:
                target.write(line)
            kept += 1
    if dry_run:
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(path)
    return {"kept": kept, "dropped": dropped}


def install_identity(*, dry_run: bool) -> int:
    records = []
    for question, answer in IDENTITY_PAIRS:
        records.append(
            {
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "domain": "general",
                "source": "identity",
                "license": "project",
            }
        )
    if dry_run:
        return len(records)
    IDENTITY_FILE.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    with APPEND_TO.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "sft_general.jsonl",
            "sft_tool.jsonl",
            "grpo_general_train.jsonl",
            "grpo_tool_train.jsonl",
            "eval_general.jsonl",
            "eval_tool.jsonl",
            "opd_prompts_general.jsonl",
            "opd_prompts_tool.jsonl",
            "opd_prompts_mixed.jsonl",
        ],
    )
    args = parser.parse_args(argv)

    for name in args.files:
        path = DATA / name
        if not path.exists():
            print(f"{name}: missing, skipped")
            continue
        stats = scrub(path, dry_run=args.dry_run)
        print(f"{name}: kept={stats['kept']} dropped={stats['dropped']}")

    added = install_identity(dry_run=args.dry_run)
    print(f"identity: {added} records -> {IDENTITY_FILE.name} (+ appended to sft_general)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
