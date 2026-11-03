"""Tokenizer fertility and corpus statistics.

Fertility is tokens per byte (and per word) by content category.  It is used to
detect a category that the MiniMind vocabulary shreds (for example formulas or
code identifiers) before spending training compute on it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..tokenizer import MiniDeepSeekTokenizer, load_tokenizer

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"\S+")
_CODE_HINTS = ("def ", "class ", "function ", "import ", "return ", "```", "{", "};")
_MATH_HINTS = ("$", "\\frac", "\\sum", "\\int", "=", "\\begin", "\\end", "∑", "√")


def categorize(text: str) -> str:
    sample = text[:4000]
    cjk = len(_CJK_RE.findall(sample))
    latin = len(_LATIN_RE.findall(sample))
    total = max(len(sample), 1)
    if any(hint in sample for hint in _CODE_HINTS) and sample.count("\n") > 2:
        return "code"
    if any(hint in sample for hint in _MATH_HINTS) and any(
        operator in sample for operator in ("=", "+", "-", "\\frac", "∑")
    ):
        return "math"
    if cjk / total > 0.2:
        return "zh"
    if latin / total > 0.5:
        return "en"
    return "other"


@dataclass
class CategoryStats:
    documents: int = 0
    chars: int = 0
    bytes_utf8: int = 0
    words: int = 0
    tokens: int = 0

    def add(self, text: str, tokens: int) -> None:
        self.documents += 1
        self.chars += len(text)
        self.bytes_utf8 += len(text.encode("utf-8"))
        self.words += len(_WORD_RE.findall(text))
        self.tokens += tokens

    def as_dict(self) -> dict[str, float]:
        return {
            "documents": self.documents,
            "chars": self.chars,
            "bytes_utf8": self.bytes_utf8,
            "words": self.words,
            "tokens": self.tokens,
            "tokens_per_byte": self.tokens / self.bytes_utf8 if self.bytes_utf8 else 0.0,
            "tokens_per_word": self.tokens / self.words if self.words else 0.0,
            "chars_per_token": self.chars / self.tokens if self.tokens else 0.0,
        }


@dataclass
class FertilityReport:
    categories: dict[str, CategoryStats] = field(default_factory=dict)
    total_tokens: int = 0
    total_bytes: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "total_tokens": self.total_tokens,
            "total_bytes": self.total_bytes,
            "overall_tokens_per_byte": (
                self.total_tokens / self.total_bytes if self.total_bytes else 0.0
            ),
            "categories": {name: stats.as_dict() for name, stats in self.categories.items()},
        }


def fertility_report(
    texts: Iterable[str],
    *,
    tokenizer: MiniDeepSeekTokenizer | None = None,
    max_documents: int | None = None,
) -> FertilityReport:
    tokenizer = tokenizer or load_tokenizer()
    report = FertilityReport()
    for index, text in enumerate(texts):
        if max_documents is not None and index >= max_documents:
            break
        category = categorize(text)
        stats = report.categories.setdefault(category, CategoryStats())
        tokens = len(tokenizer.encode(text))
        stats.add(text, tokens)
        report.total_tokens += tokens
        report.total_bytes += len(text.encode("utf-8"))
    return report
