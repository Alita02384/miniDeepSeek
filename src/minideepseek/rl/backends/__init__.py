"""Rollout backends: torch (default), optional vLLM."""

from __future__ import annotations

from pathlib import Path

from ...model.wrapper import MiniDeepSeekModel
from ..rollout import TorchRolloutEngine
from .vllm import VLLMGenerator


def build_rollout_engine(
    config,
    model: MiniDeepSeekModel,
    tokenizer,
    *,
    sample_kwargs: dict | None = None,
) -> TorchRolloutEngine:
    """Build the rollout engine selected by ``config.rl.rollout_backend``.

    ``torch`` needs no extra dependency.  ``vllm`` exports the current weights
    to an HF directory and samples with vLLM; actor log-probs still come from
    the training model, so the GRPO ratio and consistency gate stay meaningful.
    """
    del sample_kwargs
    if config.rl.rollout_backend == "torch":
        return TorchRolloutEngine(model, tokenizer, thinking_ratio=config.rl.thinking_ratio)
    if config.rl.rollout_backend != "vllm":
        raise ValueError(f"unknown rollout backend {config.rl.rollout_backend!r}")

    from ...model.hf.export import export_hf

    path = Path(config.rl.vllm_model_path or Path(config.train.out_dir) / "hf_rollout")
    export_hf(
        model.state_dict(),
        path,
        flavor=model.config.flavor,
        seq_len=model.max_context_length,
        tie_word_embeddings=model.config.tie_word_embeddings,
        swiglu_limit=model.config.layers[0].moe.swiglu_limit,
        index_topk=model.current_index_topk(),
        indexer_loss_coeff=model.config.layers[-1].attention.indexer_loss_coeff,
    )
    generator = VLLMGenerator(
        path,
        tokenizer,
        tensor_parallel_size=config.rl.vllm_tensor_parallel_size,
        gpu_memory_utilization=config.rl.vllm_gpu_memory_utilization,
        max_model_len=int(config.model.seq_len),
        flavor=model.config.flavor,
        swiglu_limit=model.config.layers[0].moe.swiglu_limit,
        index_topk=model.current_index_topk(),
        indexer_loss_coeff=model.config.layers[-1].attention.indexer_loss_coeff,
    )
    return TorchRolloutEngine(model, tokenizer, generator=generator, thinking_ratio=config.rl.thinking_ratio)
