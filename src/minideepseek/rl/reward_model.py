"""Local reward-model wrapper (optional, CUDA-heavy).

Refactored from MiniMind ``trainer/trainer_utils.py::LMForRewardModel``
(Apache-2.0).  The original pipeline uses ``internlm2-1_8b-reward`` for RLAIF
prompts that have no verifiable answer.  We keep the same call convention so
the reward function can be swapped without touching the GRPO loop.

The dependency is optional: importing this module must not import
``transformers``.  A missing dependency raises a clear installation error only
when a reward model is actually requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class LocalRewardModel:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device | None = None,
        dtype: str = "float16",
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Local reward model requires `transformers`. Install it with "
                "`pip install -e '.[hf]'` or use an HTTP judge instead."
            ) from exc
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            str(model_path), torch_dtype=torch_dtype, trust_remote_code=True
        )
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages: list[dict[str, Any]], response: str) -> float:
        history_text = "\n".join(
            f"{message.get('role')}: {message.get('content', '')}"
            for message in messages[:-1]
        )
        last_query = messages[-1].get("content", "") if messages else ""
        message_context = (
            f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}"
            if history_text
            else last_query
        )
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response},
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(float(score), 3.0), -3.0)
