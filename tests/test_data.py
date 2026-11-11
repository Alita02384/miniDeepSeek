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


def test_full_packs_recut_without_moving_tokens(tmp_path):
    """``full_packs`` re-cuts the grid; the token stream must be identical."""
    source = tmp_path / "pretrain.jsonl"
    docs = [
        "Document number one is about deep learning and neural networks. " * 3,
        "Document two is a short note about cooking, ingredients and recipes. " * 2,
        "Document three covers tool use, function calling and JSON arguments. " * 4,
        "Document four is short.",
    ]
    source.write_text("\n".join(json.dumps({"text": text}) for text in docs) + "\n")
    output = tmp_path / "pretrain.npz"
    seq_len = 64
    build_pretrain_dataset([source], output, seq_len=seq_len, min_chars=8, quality_filter=False)

    plain = PackedPretrainDataset(output)
    packed = PackedPretrainDataset(output, full_packs=True)

    # Same tokens, same order: the view change must not drop or reorder any id.
    flat_plain = [int(t) for index in range(len(plain)) for t in plain[index].input_ids]
    flat_packed = [int(t) for index in range(len(packed)) for t in packed[index].input_ids]
    assert flat_plain == flat_packed
    assert packed.num_tokens == plain.num_tokens

    # Every pack but the last is exactly seq_len tokens.
    assert len(packed) == -(-packed.num_tokens // seq_len)
    for index in range(len(packed) - 1):
        assert packed[index].num_tokens == seq_len

    for index in range(len(packed)):
        batch = packed[index]
        positions = batch.positions.tolist()
        # Positions restart at every document start *and* at every cut.
        expected: list[int] = []
        for value in positions:
            expected.append(0 if value == 0 else expected[-1] + 1)
        assert positions == expected
        starts = [i for i, value in enumerate(positions) if value == 0]
        assert starts[0] == 0
        assert batch.cu_seqlens.tolist() == starts + [len(positions)]
        assert batch.target_mask.tolist() == [value != 0 for value in positions]


def test_sft_builder_dedups_shuffles_and_budgets_across_sources(tmp_path):
    from minideepseek.data.sft import build_sft_dataset

    rows = [
        {
            "messages": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ]
        }
        for index in range(30)
    ]
    rows.extend([rows[0]] * 5)
    source = tmp_path / "sft.jsonl"
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "sft.npz"
    meta = build_sft_dataset([source], output, seq_len=256, max_records=10, seed=7)
    stats = meta["source_stats"][0]
    assert stats["total"] == 35
    assert stats["unique"] == 30
    assert stats["duplicates"] == 5
    assert meta["examples"] == 10
    assert meta["selected_records"] == 10
    assert meta["selection_sha256"]

    other = tmp_path / "sft2.npz"
    meta2 = build_sft_dataset([source], other, seq_len=256, max_records=10, seed=8)
    assert meta2["selection_sha256"] != meta["selection_sha256"]


def test_sft_mixer_weights_are_loss_token_weights(tmp_path):
    from minideepseek.config import (
        DataSettings,
        DataSourceSettings,
        TrainConfig,
        TrainSettings,
    )
    from minideepseek.data.sft import build_sft_dataset
    from minideepseek.train.run import load_dataset

    def write(path, assistant_repeat):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a" * assistant_repeat},
                ]
            }
            for _ in range(4)
        ]
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        build_sft_dataset([path], path.with_suffix(".npz"), seq_len=256)

    general = tmp_path / "general.jsonl"
    tool = tmp_path / "tool.jsonl"
    write(general, 200)
    write(tool, 20)
    config = TrainConfig(
        train=TrainSettings(steps=1, grad_accum=1),
        data=DataSettings(
            sources=[
                DataSourceSettings(path=str(general.with_suffix(".npz")), domain="general", weight=0.7),
                DataSourceSettings(path=str(tool.with_suffix(".npz")), domain="tool", weight=0.3),
            ]
        ),
    )
    dataset, _factory = load_dataset(config, "sft")
    meta = dataset.meta["sources"]
    assert meta[0]["loss_token_ratio"] > meta[1]["loss_token_ratio"]
    assert meta[0]["effective_weight"] == pytest.approx(
        0.7 / meta[0]["loss_token_ratio"]
    )
    assert meta[1]["effective_weight"] == pytest.approx(
        0.3 / meta[1]["loss_token_ratio"]
    )


def test_sft_dataset_validation_rejects_broken_masks(tmp_path):
    import numpy as np

    from minideepseek.config import TrainConfig
    from minideepseek.data.io import save_flat_dataset
    from minideepseek.data.sft import SFTDataset
    from minideepseek.tokenizer import load_tokenizer
    from minideepseek.train.run import _validate_dataset

    tokenizer = load_tokenizer()
    path = tmp_path / "broken.npz"
    save_flat_dataset(
        path,
        input_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
        positions=np.arange(4, dtype=np.int32),
        target_mask=np.zeros(4, dtype=bool),
        offsets=np.asarray([0, 4], dtype=np.int64),
        meta={
            "format": "minideepseek-sft-v1",
            "seq_len": 256,
            "tokenizer_hash": tokenizer.sha256,
        },
    )
    with pytest.raises(ValueError, match="target tokens"):
        _validate_dataset(SFTDataset(path), tokenizer, TrainConfig(), label="train")


def test_sft_require_tools_filters_the_appended_general_slice(tmp_path):
    from minideepseek.data.sft import build_sft_dataset

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "calc",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        }
    ]
    rows = [
        {
            "messages": [
                {"role": "system", "content": "", "tools": tools},
                {"role": "user", "content": "1+1"},
                {"role": "assistant", "content": "2"},
            ]
        }
        for _ in range(3)
    ]
    rows.append(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    )
    source = tmp_path / "sft_tool.jsonl"
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    meta = build_sft_dataset(
        [source],
        tmp_path / "tool.npz",
        seq_len=256,
        max_records=10,
        require_tools=True,
    )
    assert meta["examples"] == 1  # three identical tool records dedup to one
    assert meta["source_stats"][0]["filtered_no_tools"] == 1
    assert meta["require_tools"] is True
