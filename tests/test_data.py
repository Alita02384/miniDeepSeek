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
from minideepseek.data.quality import QualityConfig, score_text
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


def test_grpo_top_level_string_tools_are_parsed(tmp_path):
    from minideepseek.data.grpo import convert_record, load_records

    tools = json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "evaluate",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    record = convert_record(
        {
            "prompt": [{"role": "user", "content": "1+1"}],
            "tools": tools,
            "gt": "2",
            "domain": "math",
        },
        default_domain="math",
        default_verifier="math_exact_match",
    )
    assert [tool["function"]["name"] for tool in record.tools] == ["calc"]

    path = tmp_path / "tools.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "1+1"}],
                "verifier": "math_exact_match",
                "domain": "math",
                "tools": tools,
            }
        )
        + "\n"
    )
    loaded = load_records(path)
    assert [tool["function"]["name"] for tool in loaded[0].tools] == ["calc"]
    rendered = load_tokenizer().render_chat(
        loaded[0].prompt,
        add_generation_prompt=True,
        tools=loaded[0].resolved_tools(),
        open_thinking=True,
    )
    assert "calc" in rendered


def test_tokenize_prompts_matches_rollout_format():
    from minideepseek.data.grpo import GRPORecord, tokenize_prompts

    tokenizer = load_tokenizer()
    record = GRPORecord(
        prompt=[{"role": "user", "content": "1+1"}],
        domain="math",
        reference_answer="2",
        verifier="math_exact_match",
        tools=[
            {
                "type": "function",
                "function": {"name": "calc", "parameters": {"type": "object"}},
            }
        ],
    )
    decoded = tokenizer.decode(tokenize_prompts([record], tokenizer)[0])
    assert "calc" in decoded
    assert "<think>" in decoded
    assert "</think>" not in decoded  # thinking stays open for the rollout


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


def test_batch_token_warmup_schedule():
    from minideepseek.data.pretrain import token_schedule

    assert token_schedule(100, None, 0.0, 10, 0) == 100
    assert token_schedule(100, 400, 0.5, 10, 0) == 100
    assert token_schedule(100, 400, 0.5, 10, 5) == 400
    assert token_schedule(100, 400, 0.5, 10, 3) == 280


def test_paragraph_dedup(tmp_path):
    repeated = (
        "This paragraph is long enough to be deduplicated across the document. " * 3
    )
    text = repeated + "\n\n" + "A unique second paragraph about a different topic. " * 3
    text = text + "\n\n" + repeated
    source = tmp_path / "paragraphs.jsonl"
    source.write_text(json.dumps({"text": text}) + "\n")
    with_dedup = build_pretrain_dataset(
        [source],
        tmp_path / "a.npz",
        seq_len=256,
        min_chars=16,
        quality_filter=False,
        dedup=False,
        paragraph_dedup=True,
    )
    without = build_pretrain_dataset(
        [source],
        tmp_path / "b.npz",
        seq_len=256,
        min_chars=16,
        quality_filter=False,
        dedup=False,
        paragraph_dedup=False,
    )
    assert with_dedup["paragraphs_removed"] >= 1
    assert with_dedup["tokens"] < without["tokens"]


def test_extended_quality_filters():
    assert "binary" in score_text("valid text\x00with nul").reasons
    vendored = "node_modules/react/index.js\n" + "export const x = 1;\n" * 20
    assert "vendored_code" in score_text(vendored, domain="code").reasons
    math = "Here is a proof without any formulas. " * 20
    config = QualityConfig(min_math_formula_ratio=1e-6, require_math_answer=True)
    reasons = score_text(math, domain="math", config=config).reasons
    assert "low_formula_density" in reasons
    assert "missing_answer_marker" in reasons
    good_math = "\\boxed{42} The value satisfies $x = y + z$ and equals 42."
    good_reasons = score_text(good_math, domain="math", config=config).reasons
    assert "low_formula_density" not in good_reasons
    assert "missing_answer_marker" not in good_reasons


def test_global_batch_tokens_are_split_by_world():
    from minideepseek.train.run import _per_rank_tokens

    assert _per_rank_tokens(256, 1) == 256
    assert _per_rank_tokens(256, 2) == 128
    assert _per_rank_tokens(3, 8) == 1
    assert _per_rank_tokens(None, 8) is None


def test_flat_dataset_sidecars_are_memmapped(tmp_path):
    import numpy as np

    from minideepseek.data.io import load_flat_dataset, save_flat_dataset

    path = tmp_path / "data.npz"
    save_flat_dataset(
        path,
        input_ids=np.arange(16, dtype=np.int32),
        positions=np.arange(16, dtype=np.int32),
        target_mask=np.ones(16, dtype=bool),
        offsets=np.asarray([0, 16], dtype=np.int64),
        meta={"format": "test"},
    )
    arrays = load_flat_dataset(path)
    assert all(isinstance(array, np.memmap) for array in arrays)
    assert arrays[0].dtype == np.int32


def test_batch_token_warmup_persists_across_loader_restarts(tmp_path):
    import numpy as np

    from minideepseek.data.io import save_flat_dataset
    from minideepseek.data.pretrain import BatchTokenSchedule, PackedPretrainDataset, iter_batches
    from minideepseek.data.sft import SFTDataset
    from minideepseek.data.sft import iter_batches as sft_iter_batches

    ids = np.arange(1, 33, dtype=np.int32)  # 4 examples x 8 tokens
    positions = np.tile(np.arange(8, dtype=np.int32), 4)
    mask = np.tile(np.asarray([False] + [True] * 7), 4)
    offsets = np.arange(0, 33, 8, dtype=np.int64)
    path = tmp_path / "tiny.npz"
    save_flat_dataset(
        path,
        input_ids=ids,
        positions=positions,
        target_mask=mask,
        offsets=offsets,
        meta={"format": "test"},
    )

    def run_epoch(iterator):
        return [batch.num_tokens for batch in iterator]

    pretrain = PackedPretrainDataset(path)
    schedule = BatchTokenSchedule(base=8, final=32, warmup_ratio=0.5, total_steps=4)
    kwargs = dict(
        batch_tokens=8,
        batch_tokens_final=32,
        batch_warmup_ratio=0.5,
        total_steps=4,
        seq_len=8,
        shuffle=False,
        schedule=schedule,
    )
    first = run_epoch(iter_batches(pretrain, **kwargs))
    second = run_epoch(iter_batches(pretrain, **kwargs))
    assert first[0] == 8
    assert second[0] > first[0]  # the warmup continued instead of restarting

    sft = SFTDataset(path)
    sft_schedule = BatchTokenSchedule(base=8, final=32, warmup_ratio=0.5, total_steps=4)
    sft_kwargs = dict(
        batch_tokens=8,
        batch_tokens_final=32,
        batch_warmup_ratio=0.5,
        total_steps=4,
        shuffle=False,
        schedule=sft_schedule,
    )
    sft_first = run_epoch(sft_iter_batches(sft, **sft_kwargs))
    sft_second = run_epoch(sft_iter_batches(sft, **sft_kwargs))
    assert sft_first[0] == 8
    assert sft_second[0] > sft_first[0]


def test_sft_tool_record_keeps_tool_schema(tmp_path):
    from minideepseek.data.sft import SFTDataset, build_sft_dataset, encode_sft_record

    tools = json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_exchange_rate",
                    "description": "query rate",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    tool_calls = json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_exchange_rate",
                    "arguments": "{\"from\": \"USD\"}",
                },
            }
        ]
    )
    record = {
        "conversations": [
            {"role": "system", "content": "", "tools": tools},
            {"role": "user", "content": "rate?"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
            {"role": "tool", "content": "7.2"},
            {"role": "assistant", "content": "7.2"},
        ]
    }
    tokenizer = load_tokenizer()
    encoded = encode_sft_record(record, tokenizer, seq_len=2048)
    assert encoded is not None
    text = tokenizer.decode(encoded[0])
    assert "# Tools" in text
    assert "get_exchange_rate" in text

    source = tmp_path / "tool.jsonl"
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    output = tmp_path / "tool.npz"
    meta = build_sft_dataset([source], output, seq_len=2048)
    assert meta["examples"] == 1
    assert len(SFTDataset(output)) == 1


def test_reference_leak_does_not_reject_question_digits():
    from minideepseek.data.grpo import validate_record

    record = GRPORecord(
        prompt=[{"role": "user", "content": "What is 1+2?"}],
        reference_answer="2",
        domain="math",
        verifier="math_exact_match",
    )
    validate_record(record, load_tokenizer())
