from __future__ import annotations

import json

from minideepseek.data.sft import SFTDataset, build_sft_dataset
from minideepseek.tokenizer import assistant_token_mask, load_tokenizer, normalize_conversation


def test_tokenizer_contract():
    tokenizer = load_tokenizer()
    assert tokenizer.vocab_size == 6400
    assert tokenizer.eos_token_id == 2
    assert tokenizer.bos_token_id == 1
    assert tokenizer.pad_token_id == 0
    assert tokenizer.special_id("think_start") is not None
    assert len(tokenizer.sha256) == 64
    assert tokenizer.encode("hello world")


def test_chat_template_and_response_mask():
    tokenizer = load_tokenizer()
    messages = [
        {"role": "user", "content": "1 + 1 = ?"},
        {"role": "assistant", "content": "2", "reasoning_content": "两数相加"},
    ]
    rendered = tokenizer.render_chat(messages)
    assert rendered.startswith("<|im_start|>user")
    assert "<|im_start|>assistant" in rendered
    assert "<think>" in rendered and "</think>" in rendered
    encoded = tokenizer.encode_chat(messages, max_length=512)
    assert len(encoded.input_ids) == len(encoded.target_mask)
    assert sum(encoded.target_mask) > 0
    # The user turn must never be supervised.
    first_assistant = encoded.input_ids.index(tokenizer.special_id("think_start"))
    assert not any(encoded.target_mask[: first_assistant - 4])
    decoded = tokenizer.decode(encoded.input_ids)
    assert "1 + 1 = ?" in decoded


def test_assistant_token_mask_finds_multiple_turns():
    tokenizer = load_tokenizer()
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    encoded = tokenizer.encode_chat(messages, max_length=512)
    starts = [
        index
        for index, token in enumerate(encoded.input_ids)
        if token == tokenizer.special_id("think_start")
    ]
    assert len(starts) == 2
    assert encoded.target_mask[starts[0]] and encoded.target_mask[starts[1]]


def test_assistant_token_mask_helper():
    header = [10, 11]
    footer = [20, 21]
    ids = [0, 10, 11, 5, 6, 20, 21, 10, 11, 7]
    mask = assistant_token_mask(ids, header, footer)
    assert mask == [False, False, False, True, True, True, True, False, False, True]


def test_normalize_conversation_accepts_minimind_schema():
    messages = normalize_conversation(
        {"conversations": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}
    )
    assert messages[0]["role"] == "user"
    assert messages[1]["content"] == "y"


def test_sft_dataset_builds_response_only_targets(tmp_path):
    source = tmp_path / "sft.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4", "reasoning_content": "simple"},
            ],
            "domain": "math",
        },
        {
            "messages": [
                {"role": "user", "content": "Say hi"},
                {"role": "assistant", "content": "hi"},
            ]
        },
    ]
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    output = tmp_path / "sft.npz"
    meta = build_sft_dataset([source], output, seq_len=256)
    assert meta["examples"] == 2
    dataset = SFTDataset(output)
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.num_tokens == len(sample.input_ids)
    assert int(sample.target_mask.sum()) > 0
    # Every target position must come after a user/assistant header, not in the user text.
    assert sample.target_mask[0].item() is False
