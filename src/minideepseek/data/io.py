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


def _sidecar_path(path: str | Path, name: str) -> Path:
    path = Path(path)
    return path.with_suffix(f".{name}.npy")


def save_flat_dataset(
    path: str | Path,
    *,
    input_ids: np.ndarray,
    positions: np.ndarray,
    target_mask: np.ndarray,
    offsets: np.ndarray,
    meta: dict[str, Any],
) -> Path:
    """Save decoded arrays as memmappable ``.npy`` sidecars plus JSON meta.

    ``.npz`` members cannot be memory-mapped through ``np.load`` (even when
    uncompressed), which would force a 6B-token corpus into RAM.  The logical
    dataset path is kept for configs; the arrays live next to it as
    ``<stem>.input_ids.npy`` etc.  ``load_flat_dataset`` still reads legacy
    ``.npz`` files for existing debug data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(_sidecar_path(path, "input_ids"), np.asarray(input_ids, dtype=np.int32))
    np.save(_sidecar_path(path, "positions"), np.asarray(positions, dtype=np.int32))
    np.save(_sidecar_path(path, "target_mask"), np.asarray(target_mask, dtype=bool))
    np.save(_sidecar_path(path, "offsets"), np.asarray(offsets, dtype=np.int64))
    path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    return path


def load_flat_dataset(path: str | Path):
    """Load a flat dataset without copying the arrays into RAM.

    New datasets are a set of ``.npy`` sidecars loaded with ``mmap_mode="r"``.
    Legacy ``.npz`` files are still read (they decompress into memory, so they
    are only suitable for the small debug corpora).
    """
    path = Path(path)
    names = ("input_ids", "positions", "target_mask", "offsets")
    sidecars = [_sidecar_path(path, name) for name in names]
    if all(sidecar.exists() for sidecar in sidecars):
        arrays = [np.load(sidecar, mmap_mode="r") for sidecar in sidecars]
    else:
        if not path.exists():
            expected = ", ".join(str(sidecar) for sidecar in sidecars)
            raise FileNotFoundError(
                f"flat dataset not found: {path} (expected sidecars: {expected})"
            )
        archive = np.load(path, mmap_mode="r")
        arrays = [archive[name] for name in names]
    return tuple(arrays)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
