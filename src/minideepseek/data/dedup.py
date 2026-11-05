"""Near-duplicate detection with MinHash-LSH.

The roadmap requires document- and paragraph-level deduplication before
tokenization.  This module implements document-level MinHash-LSH plus a
streaming paragraph deduper; both are wired into the pretrain pipeline.  They
are compact and dependency-free, suitable for local debug corpora, not a
replacement for a distributed dedup pipeline at 6B tokens.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_MERSENNE = (1 << 61) - 1


def normalize_for_dedup(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def shingles(text: str, n: int = 5) -> set[int]:
    words = normalize_for_dedup(text).split()
    if len(words) < n:
        return {_hash64(" ".join(words))} if words else set()
    return {_hash64(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)}


def _hash64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


@dataclass
class MinHashLSH:
    num_hashes: int = 64
    bands: int = 8
    shingle_size: int = 5
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.num_hashes % self.bands != 0:
            raise ValueError("num_hashes must be divisible by bands")
        self.rows = self.num_hashes // self.bands
        self._params = self._make_permutations()

    def _make_permutations(self) -> list[tuple[int, int]]:
        # Deterministic, independent-enough affine permutations modulo 2^61-1.
        params = []
        for index in range(self.num_hashes):
            a = _hash64(f"a:{self.seed}:{index}") % (_MERSENNE - 1) + 1
            b = _hash64(f"b:{self.seed}:{index}") % _MERSENNE
            params.append((a, b))
        return params

    def signature(self, text: str) -> tuple[int, ...]:
        hashes = shingles(text, self.shingle_size)
        if not hashes:
            return tuple([0] * self.num_hashes)
        values = []
        for a, b in self._params:
            values.append(min(((a * h + b) % _MERSENNE) for h in hashes))
        return tuple(values)

    def duplicate_pairs(
        self, texts: list[str], *, threshold: float = 0.8
    ) -> list[tuple[int, int, float]]:
        signatures = [self.signature(text) for text in texts]
        buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        pairs: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int]] = set()
        shingle_sets = [shingles(text, self.shingle_size) for text in texts]
        for index, signature in enumerate(signatures):
            for band in range(self.bands):
                start = band * self.rows
                key = (band, signature[start : start + self.rows])
                for other in buckets.get(key, ()):
                    pair = (other, index)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    score = jaccard(shingle_sets[other], shingle_sets[index])
                    if score >= threshold:
                        pairs.append((other, index, score))
                buckets.setdefault(key, []).append(index)
        return sorted(pairs)


def duplicate_indices(texts: list[str], *, threshold: float = 0.8) -> set[int]:
    """Return indices that are duplicates of an earlier document."""
    lsh = MinHashLSH()
    pairs = lsh.duplicate_pairs(texts, threshold=threshold)
    # ``pairs`` is sorted by (left, right); the later document is the one
    # removed, so every right endpoint is a duplicate of an earlier document.
    return {right for _left, right, _score in pairs}


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


class ParagraphDeduper:
    """Drop repeated paragraphs (across documents) while keeping the first."""

    def __init__(self, *, min_chars: int = 64):
        self.min_chars = max(int(min_chars), 1)
        self._seen: set[int] = set()
        self.removed = 0

    def __call__(self, text: str) -> str:
        paragraphs = _PARAGRAPH_SPLIT.split(text)
        if len(paragraphs) <= 1:
            return text
        kept: list[str] = []
        for paragraph in paragraphs:
            normalized = normalize_for_dedup(paragraph)
            if len(normalized) < self.min_chars:
                kept.append(paragraph)
                continue
            digest = _hash64(normalized)
            if digest in self._seen:
                self.removed += 1
                continue
            self._seen.add(digest)
            kept.append(paragraph)
        return "\n\n".join(kept)


class MinHashIndex:
    """Streaming near-duplicate index (band buckets + exact Jaccard verify)."""

    def __init__(
        self,
        *,
        num_hashes: int = 64,
        bands: int = 8,
        shingle_size: int = 5,
        threshold: float = 0.8,
        seed: int = 1234,
    ):
        self.lsh = MinHashLSH(
            num_hashes=num_hashes,
            bands=bands,
            shingle_size=shingle_size,
            seed=seed,
        )
        self.threshold = threshold
        self._buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        self._signatures: list[tuple[int, ...]] = []
        self._shingles: list[set[int]] = []

    def is_duplicate(self, text: str) -> bool:
        """Return True if ``text`` is a near-duplicate of an earlier document."""
        signature = self.lsh.signature(text)
        candidates: set[int] = set()
        for band in range(self.lsh.bands):
            start = band * self.lsh.rows
            candidates.update(self._buckets.get((band, signature[start : start + self.lsh.rows]), ()))
        grams = shingles(text, self.lsh.shingle_size)
        for other in candidates:
            if jaccard(grams, self._shingles[other]) >= self.threshold:
                return True
        index = len(self._signatures)
        self._signatures.append(signature)
        self._shingles.append(grams)
        for band in range(self.lsh.bands):
            start = band * self.lsh.rows
            key = (band, signature[start : start + self.lsh.rows])
            self._buckets.setdefault(key, []).append(index)
        return False
