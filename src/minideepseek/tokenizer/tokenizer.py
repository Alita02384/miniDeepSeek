"""MiniMind tokenizer wrapper (6,400-token vocabulary).

The tokenizer is a hard project invariant (roadmap section 3.4): the special
tokens ``<think>``/``</think>`` and ``<tool_call>``/``<tool_response>`` already
exist, so no vocabulary changes are made.  Every checkpoint records the SHA-256
of ``tokenizer.json`` and training/rollout share this single wrapper.

Response-only supervision is produced by locating the
``<|im_start|>assistant\\n`` header and the matching ``<|im_end|>\\n`` footer in
the token stream (the same strategy MiniMind uses).  This is independent of
Jinja whitespace and works for multi-turn and tool-call data.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jinja2
from tokenizers import Tokenizer

from ..paths import TOKENIZER_CONFIG_FILE, TOKENIZER_FILE

_SPECIALS = {
    "eot": "<|endoftext|>",
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "think_start": "<think>",
    "think_end": "</think>",
    "tool_call_start": "<tool_call>",
    "tool_call_end": "</tool_call>",
    "tool_response_start": "<tool_response>",
    "tool_response_end": "</tool_response>",
}


@dataclass(frozen=True)
class ChatEncoding:
    input_ids: list[int]
    target_mask: list[bool]

    def __len__(self) -> int:
        return len(self.input_ids)


class MiniDeepSeekTokenizer:
    """Thin wrapper over the ``tokenizers`` library plus the chat template."""

    def __init__(
        self,
        tokenizer_file: str | Path = TOKENIZER_FILE,
        config_file: str | Path = TOKENIZER_CONFIG_FILE,
        *,
        vocab_limit: int | None = None,
    ):
        self.tokenizer_file = Path(tokenizer_file)
        self.config_file = Path(config_file)
        self._vocab_limit = int(vocab_limit) if vocab_limit else None
        if self._vocab_limit is not None and self._vocab_limit < 3:
            raise ValueError("vocab_limit must be at least 3")
        self._tokenizer = Tokenizer.from_file(str(self.tokenizer_file))
        self._config = json.loads(self.config_file.read_text(encoding="utf-8"))
        self._template_source = self._config.get("chat_template")
        self._env = jinja2.Environment(
            trim_blocks=True, lstrip_blocks=True, autoescape=False
        )
        self._template = (
            self._env.from_string(self._template_source)
            if self._template_source
            else None
        )
        self._special_ids = {
            name: self._tokenizer.token_to_id(token)
            for name, token in _SPECIALS.items()
        }
        missing = [name for name, value in self._special_ids.items() if value is None]
        if missing:
            raise ValueError(f"tokenizer is missing required special tokens: {missing}")
        self._header_ids = self.encode("<|im_start|>assistant\n")
        self._footer_ids = self.encode("<|im_end|>\n")

    # -- properties -----------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        size = int(self._tokenizer.get_vocab_size())
        return min(size, self._vocab_limit) if self._vocab_limit else size

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.tokenizer_file.read_bytes()).hexdigest()

    @property
    def eos_token_id(self) -> int:
        return int(self._special_ids["im_end"])

    @property
    def bos_token_id(self) -> int:
        return int(self._special_ids["im_start"])

    @property
    def pad_token_id(self) -> int:
        return int(self._special_ids["eot"])

    @property
    def stop_token_ids(self) -> list[int]:
        return [self.eos_token_id, self.pad_token_id]

    def special_id(self, name: str) -> int:
        return int(self._special_ids[name])

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    # -- plain text -----------------------------------------------------------
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
        if self._vocab_limit is not None:
            ids = [token % self._vocab_limit for token in ids]
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    # -- chat -----------------------------------------------------------------
    def render_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
        open_thinking: bool = False,
    ) -> str:
        if self._template is None:
            raise RuntimeError("tokenizer_config.json has no chat_template")
        return self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            open_thinking=open_thinking,
        )

    def encode_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
        max_length: int | None = None,
    ) -> ChatEncoding:
        """Render and tokenize a conversation with a response-only mask.

        ``target_mask[t]`` is true for assistant tokens (including the
        ``<think>`` scaffolding, tool calls, and the ``<|im_end|>`` footer) and
        false for system/user/tool tokens.
        """
        text = self.render_chat(
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        ids = self.encode(text)
        mask = assistant_token_mask(ids, self._header_ids, self._footer_ids)
        if max_length is not None and len(ids) > max_length:
            ids, mask = _truncate_right(ids, mask, max_length)
        return ChatEncoding(input_ids=ids, target_mask=mask)

    def assistant_prefix_ids(self) -> list[int]:
        return list(self._header_ids)

    def assistant_footer_ids(self) -> list[int]:
        return list(self._footer_ids)


def assistant_token_mask(
    ids: list[int],
    header_ids: list[int],
    footer_ids: list[int],
) -> list[bool]:
    """Mark tokens inside ``<|im_start|>assistant\\n ... <|im_end|>\\n`` spans."""
    mask = [False] * len(ids)
    i = 0
    n_header, n_footer = len(header_ids), len(footer_ids)
    while i < len(ids):
        if ids[i : i + n_header] == header_ids:
            start = i + n_header
            end = start
            while end < len(ids) and ids[end : end + n_footer] != footer_ids:
                end += 1
            stop = min(end + n_footer, len(ids)) if end < len(ids) else end
            for j in range(start, stop):
                mask[j] = True
            i = stop if stop > i else i + 1
        else:
            i += 1
    return mask


def _truncate_right(
    ids: list[int], mask: list[bool], max_length: int
) -> tuple[list[int], list[bool]]:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    return ids[:max_length], mask[:max_length]


def normalize_conversation(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept both MiniMind (``conversations``) and roadmap (``messages``) schemas."""
    messages = item.get("messages") or item.get("conversations")
    if not isinstance(messages, list) or not messages:
        raise ValueError("record has no non-empty 'messages'/'conversations' list")
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or "role" not in message:
            raise ValueError(f"invalid message: {message!r}")
        role = str(message["role"])
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        entry: dict[str, Any] = {"role": role, "content": content}
        reasoning = message.get("reasoning_content")
        if reasoning:
            entry["reasoning_content"] = str(reasoning)
        tools = _parse_json_field(message.get("tools"), "tools")
        if tools:
            entry["tools"] = tools
        tool_calls = _parse_json_field(message.get("tool_calls"), "tool_calls")
        if tool_calls:
            entry["tool_calls"] = tool_calls
        normalized.append(entry)
    return normalized


def _parse_json_field(value: Any, field: str) -> Any:
    """MiniMind stores ``tools``/``tool_calls`` as JSON strings."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid {field} JSON: {error}") from error
    if field == "tool_calls" and not isinstance(value, list):
        raise ValueError("tool_calls must be a list")
    if field == "tools" and not isinstance(value, list):
        raise ValueError("tools must be a list")
    return value


@lru_cache(maxsize=8)
def _load_tokenizer_cached(
    tokenizer_file: str | None = None,
    config_file: str | None = None,
    vocab_limit: int | None = None,
) -> MiniDeepSeekTokenizer:
    return MiniDeepSeekTokenizer(
        tokenizer_file or TOKENIZER_FILE,
        config_file or TOKENIZER_CONFIG_FILE,
        vocab_limit=vocab_limit,
    )


def load_tokenizer(
    tokenizer_file: str | None = None,
    config_file: str | None = None,
    *,
    vocab_limit: int | None = None,
) -> MiniDeepSeekTokenizer:
    """Load the tokenizer.

    ``vocab_limit`` remaps every token id modulo the limit.  This exists only
    for the 2048-vocabulary debug model; it is lossy and must never be used for
    real training.  It can also be selected with ``MINIDEEPSEEK_VOCAB_LIMIT``.
    """
    if vocab_limit is None:
        raw = os.environ.get("MINIDEEPSEEK_VOCAB_LIMIT")
        vocab_limit = int(raw) if raw else None
    return _load_tokenizer_cached(tokenizer_file, config_file, vocab_limit)
