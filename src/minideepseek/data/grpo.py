"""GRPO prompt data: schema, validation, conversion, and splits.

The prompt is the only model-visible field.  Reference answers, rubrics and
verifier names go to the reward layer and must never be rendered into the
prompt (roadmap section 4.4).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..tokenizer import MiniDeepSeekTokenizer, load_tokenizer, normalize_conversation
from .io import iter_jsonl, write_jsonl

VERIFIERS = {"math_exact_match", "unit_test", "llm_judge", "numeric", "none"}
DOMAINS = {"math", "code", "world", "medical", "general"}


@dataclass
class GRPORecord:
    prompt: list[dict[str, Any]]
    domain: str = "general"
    reference_answer: str | None = None
    rubric: dict[str, Any] = field(default_factory=dict)
    verifier: str = "none"
    difficulty: str = "medium"
    source: str = "unknown"
    license: str = "unknown"
    tools: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "domain": self.domain,
            "reference_answer": self.reference_answer,
            "rubric": self.rubric,
            "verifier": self.verifier,
            "difficulty": self.difficulty,
            "source": self.source,
            "license": self.license,
            **({"tools": self.tools} if self.tools else {}),
        }

    def resolved_tools(self) -> list[dict[str, Any]] | None:
        """Return top-level tools or the system-message tool schema."""
        tools = self.tools
        if not tools:
            for message in self.prompt:
                tools = message.get("tools")
                if tools:
                    break
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                return None
        return tools if isinstance(tools, list) else None


def validate_record(record: GRPORecord, tokenizer: MiniDeepSeekTokenizer) -> None:
    if not record.prompt:
        raise ValueError("GRPO prompt is empty")
    if record.verifier not in VERIFIERS:
        raise ValueError(f"unknown verifier {record.verifier!r}")
    if record.domain not in DOMAINS:
        raise ValueError(f"unknown domain {record.domain!r}")
    rendered = tokenizer.render_chat(
        record.prompt, add_generation_prompt=True, tools=record.resolved_tools()
    )
    reference = record.reference_answer
    if reference and reference.strip() and reference.strip() in rendered:
        # A reference answer leaking into the prompt is the classic GRPO bug.
        raise ValueError("reference answer appears in the rendered prompt")
    for message in record.prompt:
        if message.get("role") == "assistant":
            raise ValueError("GRPO prompts must not contain assistant turns")


def _prompt_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = record.get("prompt") or record.get("messages")
    if explicit is not None:
        messages = normalize_conversation({"messages": explicit})
    elif record.get("conversations"):
        messages = normalize_conversation(record)
    elif isinstance(record.get("question"), str):
        messages = [{"role": "user", "content": record["question"]}]
    else:
        raise ValueError("cannot build a prompt: no prompt/messages/conversations/question")
    # Keep the turns up to (and including) the last user message; a trailing
    # assistant turn becomes the reference instead.
    while messages and messages[-1]["role"] != "user":
        messages = messages[:-1]
    if not messages:
        messages = normalize_conversation({"messages": explicit or []})
    for message in messages:
        if message["role"] == "assistant":
            raise ValueError("prompt contains an assistant turn")
    return messages


def _reference_from_record(record: dict[str, Any], messages: list[dict[str, Any]]) -> str | None:
    for key in ("reference_answer", "answer", "gt", "target"):
        value = record.get(key)
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        if isinstance(value, str) and value.strip():
            return value.strip()
    original = record.get("conversations")
    if isinstance(original, list):
        for message in reversed(original):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return None


def convert_record(
    record: dict[str, Any],
    *,
    default_domain: str = "general",
    default_verifier: str = "none",
    source: str | None = None,
) -> GRPORecord:
    prompt = _prompt_from_record(record)
    domain = str(record.get("domain") or default_domain)
    verifier = str(record.get("verifier") or default_verifier)
    return GRPORecord(
        prompt=prompt,
        domain=domain if domain in DOMAINS else default_domain,
        reference_answer=_reference_from_record(record, prompt),
        rubric=dict(record.get("rubric") or {}),
        verifier=verifier if verifier in VERIFIERS else default_verifier,
        difficulty=str(record.get("difficulty") or "medium"),
        source=str(record.get("source") or source or "unknown"),
        license=str(record.get("license") or "unknown"),
        tools=list(record.get("tools") or []),
    )


def load_records(path: str | Path) -> list[GRPORecord]:
    path = Path(path)
    records = []
    for item in iter_jsonl(path):
        if "prompt" in item and "verifier" in item:
            records.append(
                GRPORecord(
                    prompt=normalize_conversation({"messages": item["prompt"]}),
                    domain=str(item.get("domain", "general")),
                    reference_answer=item.get("reference_answer"),
                    rubric=dict(item.get("rubric") or {}),
                    verifier=str(item.get("verifier", "none")),
                    difficulty=str(item.get("difficulty", "medium")),
                    source=str(item.get("source", "unknown")),
                    license=str(item.get("license", "unknown")),
                    tools=list(item.get("tools") or []),
                )
            )
        else:
            records.append(convert_record(item))
    return records


def build_grpo_dataset(
    inputs: list[str | Path],
    output_path: str | Path,
    *,
    domain: str = "math",
    verifier: str = "math_exact_match",
    tokenizer: MiniDeepSeekTokenizer | None = None,
    require_reference: bool = True,
    max_records: int | None = None,
    seed: int = 1234,
) -> dict[str, Any]:
    tokenizer = tokenizer or load_tokenizer()
    records: list[GRPORecord] = []
    skipped = 0
    for path in inputs:
        for item in iter_jsonl(path):
            if max_records is not None and len(records) >= max_records:
                break
            try:
                record = convert_record(
                    item,
                    default_domain=domain,
                    default_verifier=verifier,
                    source=Path(path).stem,
                )
                if require_reference and not record.reference_answer:
                    skipped += 1
                    continue
                validate_record(record, tokenizer)
            except ValueError:
                skipped += 1
                continue
            records.append(record)
        if max_records is not None and len(records) >= max_records:
            break
    if not records:
        raise ValueError("no valid GRPO records were produced")
    write_jsonl(output_path, (record.to_dict() for record in records))
    return {
        "format": "minideepseek-grpo-v1",
        "records": len(records),
        "skipped": skipped,
        "domain": domain,
        "verifier": verifier,
        "tokenizer_hash": tokenizer.sha256,
        "sources": [str(Path(p)) for p in inputs],
        "seed": int(seed),
    }


def split_records(
    records: list[GRPORecord], *, val_ratio: float = 0.1, seed: int = 1234
) -> tuple[list[GRPORecord], list[GRPORecord]]:
    import random

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    n_val = int(len(records) * val_ratio)
    val_indices = set(order[:n_val])
    train = [record for i, record in enumerate(records) if i not in val_indices]
    val = [record for i, record in enumerate(records) if i in val_indices]
    return train, val


def tokenize_prompts(
    records: Iterable[GRPORecord],
    tokenizer: MiniDeepSeekTokenizer,
) -> list[list[int]]:
    prompts = []
    for record in records:
        text = tokenizer.render_chat(record.prompt, add_generation_prompt=True)
        prompts.append(tokenizer.encode(text))
    return prompts
