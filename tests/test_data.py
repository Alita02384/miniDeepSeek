from __future__ import annotations

import json

import pytest

from minideepseek.data.contamination import build_eval_ngrams, find_contamination
from minideepseek.data.dedup import MinHashIndex, duplicate_indices, jaccard, shingles
from minideepseek.data.fertility import categorize, fertility_report
from minideepseek.data.grpo import GRPORecord, build_grpo_dataset, convert_record, validate_record
from minideepseek.data.manifest import DataManifest, SourceSpec, validate_manifest
from minideepseek.data.opd import balance_by_teacher, build_opd_pool, load_opd_prompts
from minideepseek.data.pretrain import PackedPretrainDataset, build_pretrain_dataset
from minideepseek.data.quality import score_text
from minideepseek.tokenizer import load_tokenizer


def test_pretrain_pack_and_filter(tmp_path):
    source = tmp_path / "pretrain.jsonl"
    docs = [
        "The first document talks about deep learning and neural networks in detail. " * 8,
        "The first document talks about deep learning and neural networks in detail. " * 8,  # duplicate
        "A different document about cooking, ingredients, and recipes for dinner tonight. " * 8,
        "short",
    ]
    source.write_text("\n".join(json.dumps({"text": text}) for text in docs) + "\n")
    output = tmp_path / "pretrain.npz"
    meta = build_pretrain_dataset([source], output, seq_len=64, min_chars=16, quality_filter=False)
    assert meta["tokens"] > 0
    assert meta["dropped_duplicate"] >= 1
    dataset = PackedPretrainDataset(output)
    assert len(dataset) >= 1
    batch = dataset[0]
    assert batch.positions[0].item() == 0
    assert batch.target_mask[0].item() is False
    assert batch.num_tokens <= 64


def test_minhash_duplicate_detection():
    texts = [
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
        "completely different words about a cooking recipe and dinner",
    ]
    assert duplicate_indices(texts, threshold=0.8) == {1}
    index = MinHashIndex()
    assert index.is_duplicate(texts[0]) is False
    assert index.is_duplicate(texts[1]) is True
    assert jaccard(shingles(texts[0]), shingles(texts[1])) == 1.0


def test_contamination_finds_13gram():
    eval_text = "What is the capital of France? Answer: Paris."
    train_text = "Lorem ipsum " + eval_text + " and more text."
    ngrams = build_eval_ngrams([eval_text], n=5)
    matches = find_contamination([train_text], ngrams, n=5)
    assert len(matches) == 1
    assert matches[0].overlap > 0


def test_quality_filters():
    assert score_text("short").reasons
    repeated = "line\n" * 100
    assert "repeated_lines" in score_text(repeated).reasons
    good = (
        "Deep learning models learn representations from data. "
        "The weather in Lisbon is mild in spring, while Reykjavik is colder. "
        "A good risotto needs patience, warm stock, and constant stirring. "
        "Rust prevents memory bugs by tracking ownership at compile time."
    )
    assert score_text(good).accepted
    code = "def add(a, b):\n    return a + b\n\nprint(add(1, 2))\n" * 4
    assert categorize(code) == "code"


def test_fertility_report():
    report = fertility_report(["你好世界，这是中文。", "Hello world, this is English."])
    payload = report.as_dict()
    assert payload["total_tokens"] > 0
    assert "zh" in payload["categories"]


def test_manifest_validation(tmp_path):
    source = tmp_path / "a.jsonl"
    source.write_text('{"text": "hello"}\n')
    manifest = DataManifest(
        name="test",
        seq_len=128,
        total_tokens=1000,
        sources=[
            SourceSpec(name="a", path=str(source), domain="general", weight=0.5, available_tokens=1000),
            SourceSpec(name="b", path=str(source), domain="code", weight=0.5, available_tokens=5000),
        ],
    )
    report = validate_manifest(manifest)
    assert report.ok, report.errors
    assert sum(entry.target_tokens for entry in report.entries) == 1000

    bad = DataManifest(
        name="bad",
        seq_len=128,
        total_tokens=1000,
        sources=[
            SourceSpec(name="a", path=str(source), domain="general", weight=0.6, available_tokens=10),
            SourceSpec(name="b", path=str(source), domain="code", weight=0.6, available_tokens=10),
        ],
    )
    bad_report = validate_manifest(bad)
    assert not bad_report.ok
    assert any("sum to 1.0" in error for error in bad_report.errors)


def test_grpo_conversion_and_reference_leak():
    record = convert_record(
        {
            "prompt": [{"role": "user", "content": "What is 1+1?"}],
            "reference_answer": "2",
            "domain": "math",
            "verifier": "math_exact_match",
        }
    )
    assert isinstance(record, GRPORecord)
    validate_record(record, load_tokenizer())
    leaky = GRPORecord(
        prompt=[{"role": "user", "content": "Answer is 2"}],
        reference_answer="2",
        domain="math",
        verifier="math_exact_match",
    )
    with pytest.raises(ValueError, match="reference answer"):
        validate_record(leaky, load_tokenizer())
    assistant = GRPORecord(
        prompt=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        domain="math",
        verifier="math_exact_match",
    )
    with pytest.raises(ValueError, match="assistant"):
        validate_record(assistant, load_tokenizer())


def test_grpo_build_from_minimind_style(tmp_path):
    source = tmp_path / "rl.jsonl"
    rows = [
        {
            "conversations": [
                {"role": "user", "content": "3 + 4 = ?"},
                {"role": "assistant", "content": "7"},
            ],
            "gt": ["7"],
        },
        {"conversations": [{"role": "user", "content": "no answer"}]},
    ]
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    meta = build_grpo_dataset([source], tmp_path / "out.jsonl", domain="math", verifier="math_exact_match")
    assert meta["records"] == 1
    assert meta["skipped"] >= 1


def test_opd_pool(tmp_path):
    source = tmp_path / "grpo.jsonl"
    rows = [
        {"prompt": [{"role": "user", "content": "math q"}], "teacher_key": "math", "domain": "math"}
        for _ in range(3)
    ] + [
        {"prompt": [{"role": "user", "content": "code q"}], "teacher_key": "code", "domain": "code"}
        for _ in range(2)
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "opd.jsonl"
    meta = build_opd_pool([source], output)
    assert meta["prompts"] == 5
    prompts = load_opd_prompts(output)
    assert len(prompts) == 5
    balanced = balance_by_teacher(prompts)
    assert len(balanced) == 5


def test_code_license_filter(tmp_path):
    source = tmp_path / "code.jsonl"
    docs = [
        "# SPDX-License-Identifier: GPL-3.0\nprint('copyleft')\n" * 4,
        "# SPDX-License-Identifier: MIT\ndef f():\n    return 1\n" * 4,
    ]
    source.write_text("\n".join(json.dumps({"text": text}) for text in docs) + "\n")
    output = tmp_path / "code.npz"
    meta = build_pretrain_dataset(
        [source], output, seq_len=128, min_chars=16, domain="code", quality_filter=False
    )
    assert meta["dropped_license"] >= 1
