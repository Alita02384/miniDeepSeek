"""Shared test fixtures.  Every model test uses the V4 debug model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from minideepseek.model import MiniDeepSeekModel, build_model, from_sequences
from minideepseek.model.batch import PackedBatch


class StubTokenizer:
    """A 2048-token stand-in so debug-model tests can use real generation.

    The real MiniMind tokenizer is used in data/tokenizer tests; the debug model
    only has a 2048-token vocabulary, so generation/GRPO/OPD tests use this
    reversible-by-contract stub.
    """

    vocab_size = 2048
    sha256 = "stub-tokenizer"
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    stop_token_ids = [2, 0]

    def render_chat(self, messages, *, add_generation_prompt=False, tools=None, open_thinking=False):
        del add_generation_prompt, tools, open_thinking
        return " ".join(str(message.get("content", "")) for message in messages)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = [3 + (ord(char) % 2044) for char in str(text)][:64]
        return ids or [3]

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(str(int(value)) for value in ids)


@pytest.fixture()
def stub_tokenizer() -> StubTokenizer:
    return StubTokenizer()


@pytest.fixture()
def make_debug_model():
    def _make(*, n_mtp_layers: int = 0, seq_len: int = 128, seed: int = 0, swiglu_limit: float | None = 10.0) -> MiniDeepSeekModel:
        torch.manual_seed(seed)
        return build_model(
            "debugmodel",
            seq_len=seq_len,
            n_mtp_layers=n_mtp_layers,
            device="cpu",
            swiglu_limit=swiglu_limit,
        )

    return _make


@pytest.fixture()
def debug_model(make_debug_model) -> MiniDeepSeekModel:
    return make_debug_model()


@pytest.fixture()
def mtp_debug_model(make_debug_model) -> MiniDeepSeekModel:
    return make_debug_model(n_mtp_layers=1)


@pytest.fixture()
def small_batch() -> PackedBatch:
    return from_sequences([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13]])


@pytest.fixture()
def tmp_jsonl(tmp_path: Path):
    def _write(name: str, records: list[dict]) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n")
        return path

    return _write
