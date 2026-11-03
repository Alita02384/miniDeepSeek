"""Small JSONL / text helpers shared by the data pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            yield record


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def jsonl_fingerprint(path: str | Path, *, max_bytes: int = 8 * 1024 * 1024) -> str:
    """Cheap content fingerprint (first ``max_bytes``) for manifests."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        digest.update(handle.read(max_bytes))
    return digest.hexdigest()


def default_text(record: dict[str, Any]) -> str:
    for key in ("text", "content", "raw_content", "document"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(
        "record has no text field; expected one of text/content/raw_content/document"
    )


def normalize_whitespace(text: str) -> str:
    """Normalize newlines and trailing spaces without changing lexical content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def save_flat_dataset(
    path: str | Path,
    *,
    input_ids: np.ndarray,
    positions: np.ndarray,
    target_mask: np.ndarray,
    offsets: np.ndarray,
    meta: dict[str, Any],
) -> Path:
    """Save a flat token dataset plus per-pack/per-example offsets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        input_ids=np.asarray(input_ids, dtype=np.int32),
        positions=np.asarray(positions, dtype=np.int32),
        target_mask=np.asarray(target_mask, dtype=bool),
        offsets=np.asarray(offsets, dtype=np.int64),
    )
    path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    return path


def load_flat_dataset(path: str | Path):
    """Load a flat dataset in memory-mapped mode.

    ``np.load`` with ``mmap_mode`` works for ``.npz``; the returned arrays are
    read-only memmaps, which keeps 6B-token datasets out of RAM.
    """
    path = Path(path)
    archive = np.load(path, mmap_mode="r")
    return archive["input_ids"], archive["positions"], archive["target_mask"], archive["offsets"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
