"""Hugging Face configuration for the miniDeepSeek V4 model.

The canonical model config lives in :mod:`minideepseek.model.config`; this class
is the thin HF-facing envelope that stores the flavor + build overrides so
``from_pretrained`` can rebuild the exact architecture.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

from minideepseek.model.config import build_model_config


class MiniDeepSeekConfig(PretrainedConfig):
    model_type = "minideepseek"

    def __init__(
        self,
        flavor: str = "debugmodel",
        seq_len: int = 2048,
        max_context: int | None = None,
        n_mtp_layers: int = 0,
        tie_word_embeddings: bool | None = None,
        swiglu_limit: float | None = 10.0,
        index_topk: int | None = None,
        indexer_loss_coeff: float = 0.01,
        vocab_size: int | None = None,
        **kwargs: Any,
    ):
        model_config = build_model_config(
            flavor,
            seq_len=seq_len,
            max_context=max_context,
            n_mtp_layers=n_mtp_layers,
            tie_word_embeddings=tie_word_embeddings,
            swiglu_limit=swiglu_limit,
            index_topk=index_topk,
            indexer_loss_coeff=indexer_loss_coeff,
        )
        if vocab_size is not None and int(vocab_size) != model_config.vocab_size:
            raise ValueError(
                f"vocab_size {vocab_size} does not match flavor {flavor} "
                f"({model_config.vocab_size})"
            )
        super().__init__(
            tie_word_embeddings=model_config.tie_word_embeddings,
            vocab_size=model_config.vocab_size,
            **kwargs,
        )
        self.flavor = flavor
        self.seq_len = int(seq_len)
        self.max_context = int(model_config.max_context_length)
        self.n_mtp_layers = int(n_mtp_layers)
        self.swiglu_limit = swiglu_limit
        self.index_topk = index_topk
        self.indexer_loss_coeff = float(indexer_loss_coeff)
        self.use_cache = False
        self.architectures = ["MiniDeepSeekForCausalLM"]
        self.auto_map = {
            "AutoConfig": "configuration_minideepseek.MiniDeepSeekConfig",
            "AutoModelForCausalLM": "modeling_minideepseek.MiniDeepSeekForCausalLM",
        }

    def to_model_config(self):
        return build_model_config(
            self.flavor,
            seq_len=self.seq_len,
            max_context=self.max_context,
            n_mtp_layers=self.n_mtp_layers,
            tie_word_embeddings=self.tie_word_embeddings,
            swiglu_limit=self.swiglu_limit,
            index_topk=self.index_topk,
            indexer_loss_coeff=self.indexer_loss_coeff,
        )
