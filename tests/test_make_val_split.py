"""The validation-tail carve: pack equivalence, trim safety, and provenance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from minideepseek.data.io import load_flat_dataset, save_flat_dataset

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "make_val_split", ROOT / "scripts" / "make_val_split.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def splitter():
    return _load_script()


def _dataset(tmp_path: Path, documents: int, length: int, seq_len: int) -> Path:
    rng = np.random.default_rng(0)
    ids = np.concatenate(
        [rng.integers(0, 64, size=length).astype(np.int32) for _ in range(documents)]
    )
    positions = np.concatenate(
        [np.arange(length, dtype=np.int32) for _ in range(documents)]
    )
    mask = positions != 0
    total = ids.size
    offsets = np.arange(0, total + 1, seq_len, dtype=np.int64)
    offsets = np.append(offsets[offsets < total], total)
    path = tmp_path / "src_ds.npz"
    save_flat_dataset(
        path,
        input_ids=ids,
        positions=positions,
        target_mask=mask,
        offsets=offsets,
        meta={
            "format": "minideepseek-pretrain-v1",
            "seq_len": seq_len,
            "packs": int(len(offsets) - 1),
            "tokens": int(total),
            "target_tokens": int(mask.sum()),
            "documents_kept": documents,
            "domain": "general",
            "tokenizer_hash": "deadbeef",
            "tokenizer_vocab": 64,
        },
    )
    return path


def test_pack_documents_matches_the_training_packer(splitter):
    from minideepseek.data.pretrain import pack_documents

    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
    stream = np.array([token for doc in docs for token in doc], dtype=np.int32)
    bounds = []
    cursor = 0
    for doc in docs:
        bounds.append((cursor, cursor + len(doc)))
        cursor += len(doc)
    ours = splitter.pack_documents(stream, bounds, 6)
    reference = pack_documents(docs, seq_len=6)
    for mine, theirs in zip(ours, reference, strict=True):
        assert np.array_equal(mine, theirs)


def test_document_bounds_treats_a_mid_document_cut_as_a_start(splitter):
    positions = np.array([5, 6, 7, 0, 1, 2], dtype=np.int32)
    assert splitter.document_bounds(positions) == [(0, 3), (3, 6)]


def test_write_val_split_keeps_the_source_untouched(splitter, tmp_path):
    source = _dataset(tmp_path, documents=6, length=1000, seq_len=2000)
    before = np.array(load_flat_dataset(source)[0])
    meta = splitter.write_val_split(
        source, tokens=1500, seq_len=4096, domain="general", out=tmp_path / "val.npz"
    )
    assert meta["tokens"] == 1500
    assert meta["split"] == "tail_holdout"
    assert meta["tokenizer_hash"] == "deadbeef"
    after = np.array(load_flat_dataset(source)[0])
    assert np.array_equal(before, after), "carving must not rewrite the source"


def test_trim_source_removes_only_the_tail_and_is_atomic(splitter, tmp_path):
    source = _dataset(tmp_path, documents=6, length=1000, seq_len=2000)
    before = np.array(load_flat_dataset(source)[0])
    trimmed = splitter.trim_source(source, 1500)
    ids, positions, _, offsets = load_flat_dataset(source)
    assert ids.size == before.size - 1500
    assert np.array_equal(np.array(ids), before[: ids.size]), "kept region must be verbatim"
    assert int(offsets[-1]) == ids.size
    assert trimmed["tokens"] == ids.size
    assert trimmed["max_tokens"] == ids.size
    assert json.loads(source.with_suffix(".json").read_text())["tokens"] == ids.size
    assert positions[0] == 0
    assert not [p for p in tmp_path.iterdir() if "staging" in p.name]


def test_trim_refuses_a_window_larger_than_the_dataset(splitter, tmp_path):
    source = _dataset(tmp_path, documents=2, length=100, seq_len=200)
    with pytest.raises(SystemExit):
        splitter.write_val_split(
            source, tokens=500, seq_len=128, domain="general", out=tmp_path / "v.npz"
        )
