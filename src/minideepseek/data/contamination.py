"""Evaluation-set contamination checking with 13-gram overlap.

The roadmap requires that GSM8K / MATH / HumanEval / MBPP / C-Eval questions and
answers never appear in pretraining, SFT, or GRPO data.  We use word/CJK-token
13-grams, normalize case and whitespace, and report every overlapping document.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
DEFAULT_N = 13


def tokens_for_ngram(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def ngram_set(text: str, n: int = DEFAULT_N) -> set[tuple[str, ...]]:
    tokens = tokens_for_ngram(text)
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def build_eval_ngrams(eval_texts: Iterable[str], n: int = DEFAULT_N) -> set[tuple[str, ...]]:
    result: set[tuple[str, ...]] = set()
    for text in eval_texts:
        result |= ngram_set(text, n)
    return result


@dataclass
class ContaminationMatch:
    index: int
    overlap: int
    example: tuple[str, ...]


def find_contamination(
    train_texts: Iterable[str],
    eval_ngrams: set[tuple[str, ...]],
    *,
    n: int = DEFAULT_N,
    max_reports: int = 100,
) -> list[ContaminationMatch]:
    matches: list[ContaminationMatch] = []
    for index, text in enumerate(train_texts):
        grams = ngram_set(text, n)
        overlap = grams & eval_ngrams
        if overlap:
            matches.append(
                ContaminationMatch(index=index, overlap=len(overlap), example=next(iter(overlap)))
            )
            if len(matches) >= max_reports:
                break
    return matches


def iter_eval_texts(records: Iterable[dict]) -> Iterator[str]:
    """Extract question/answer text from common eval schemas."""
    for record in records:
        parts = []
        for key in ("question", "problem", "prompt", "answer", "solution", "reference_answer"):
            value = record.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and isinstance(item.get("content"), str):
                        parts.append(item["content"])
        yield "\n".join(parts)
