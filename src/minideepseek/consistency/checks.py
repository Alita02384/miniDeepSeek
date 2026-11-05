"""Consistency checks for log-probabilities, routing, and checkpoints.

These are the gate from roadmap section 6.  They run the *same* model twice on
the same tokens through different paths (rollout vs teacher forcing, record vs
replay, save vs reload) and compare the numbers directly.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..model.batch import from_sequences
from ..model.checkpoint import load_checkpoint, save_checkpoint
from ..model.routing import (
    compare_routing,
    record_routing,
    replay_routing,
    routing_metrics,
)
from ..model.wrapper import MiniDeepSeekModel
from ..rl.rollout import RolloutSample, TorchRolloutEngine
from ..tokenizer import MiniDeepSeekTokenizer


@dataclass
class LogprobReport:
    max_abs_diff: float
    mean_abs_diff: float
    checked_tokens: int
    threshold: float
    passed: bool
    per_prompt: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "checked_tokens": self.checked_tokens,
            "threshold": self.threshold,
            "passed": self.passed,
            "per_prompt": self.per_prompt,
        }


def check_logprob_consistency(
    model: MiniDeepSeekModel,
    tokenizer: MiniDeepSeekTokenizer,
    prompts: Sequence[list[int]],
    *,
    max_new_tokens: int = 8,
    threshold: float = 1e-3,
    seed: int = 1234,
) -> LogprobReport:
    """Compare sampling-time log-probs with a teacher-forced forward pass."""
    if not prompts:
        raise ValueError("prompts must not be empty")
    engine = TorchRolloutEngine(model, tokenizer)
    output = model.generate(
        [list(prompt) for prompt in prompts],
        max_new_tokens=max_new_tokens,
        greedy=True,
        stop_token_ids=tokenizer.stop_token_ids,
        seed=seed,
    )
    diffs: list[float] = []
    per_prompt: list[float] = []
    for index, prompt in enumerate(prompts):
        sample = RolloutSample(
            prompt_ids=list(prompt),
            response_ids=list(output.response_ids[index]),
            sampling_logprobs=list(output.response_logprobs[index]),
        )
        if not sample.response_ids:
            per_prompt.append(0.0)
            continue
        actor = engine.actor_logprobs([sample], temperature=1.0).detach().float().cpu()
        sampling = torch.tensor(sample.sampling_logprobs, dtype=torch.float32)
        if actor.numel() != sampling.numel():
            raise RuntimeError(
                f"logprob length mismatch for prompt {index}: "
                f"{actor.numel()} vs {sampling.numel()}"
            )
        diff = (actor - sampling).abs()
        diffs.extend(diff.tolist())
        per_prompt.append(float(diff.max()) if diff.numel() else 0.0)
    max_diff = max(diffs) if diffs else 0.0
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    return LogprobReport(
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        checked_tokens=len(diffs),
        threshold=threshold,
        passed=max_diff <= threshold,
        per_prompt=per_prompt,
    )


@dataclass
class RoutingReport:
    match_rate: float
    per_layer: dict[str, float]
    threshold: float
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "match_rate": self.match_rate,
            "per_layer": self.per_layer,
            "threshold": self.threshold,
            "passed": self.passed,
            "metrics": self.metrics,
        }


def check_routing_consistency(
    model: MiniDeepSeekModel,
    tokenizer: MiniDeepSeekTokenizer,
    prompts: Sequence[list[int]],
    *,
    max_new_tokens: int = 8,
    threshold: float = 0.999,
    seed: int = 1234,
) -> RoutingReport:
    """Record routing during rollout, replay it in the actor, compare top-k sets."""
    engine = TorchRolloutEngine(model, tokenizer)
    output = model.generate(
        [list(prompt) for prompt in prompts],
        max_new_tokens=max_new_tokens,
        greedy=True,
        stop_token_ids=tokenizer.stop_token_ids,
        seed=seed,
    )
    samples = [
        RolloutSample(
            prompt_ids=list(prompt),
            response_ids=list(output.response_ids[index]),
            sampling_logprobs=list(output.response_logprobs[index]),
        )
        for index, prompt in enumerate(prompts)
    ]
    samples = [sample for sample in samples if sample.response_ids]
    if not samples:
        raise ValueError("no generated tokens to check")
    rollout_record = engine.record_routing_for(samples)
    batch, _ = engine.build_batch(samples, model.device)
    was_training = model.training
    model.eval()
    try:
        with (
            replay_routing(model.model, rollout_record),
            record_routing(model.model) as actor_record,
        ):
            model.forward(batch, use_mtp=False)
    finally:
        if was_training:
            model.train()
    per_layer = compare_routing(rollout_record, actor_record)
    match_rate = per_layer.pop("overall")
    return RoutingReport(
        match_rate=match_rate,
        per_layer=per_layer,
        threshold=threshold,
        passed=match_rate >= threshold,
        metrics=routing_metrics(rollout_record),
    )


@dataclass
class CheckpointReport:
    max_abs_diff: float
    passed: bool
    num_tensors: int
    checkpoint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_abs_diff": self.max_abs_diff,
            "passed": self.passed,
            "num_tensors": self.num_tensors,
            "checkpoint": self.checkpoint,
        }


@torch.no_grad()
def check_checkpoint_consistency(
    model: MiniDeepSeekModel,
    tokenizer: MiniDeepSeekTokenizer,
    prompts: Sequence[list[int]],
    *,
    directory: str | Path | None = None,
    flavor: str,
    seq_len: int,
    n_mtp_layers: int = 0,
    index_topk: int | None = None,
    tie_word_embeddings: bool | None = None,
    swiglu_limit: float | None = 10.0,
    indexer_loss_coeff: float = 0.01,
) -> CheckpointReport:
    """Save, rebuild, reload, and compare logits bit-for-bit (CPU/float32)."""
    from ..model.wrapper import build_model

    tempdir = Path(directory or tempfile.mkdtemp(prefix="minideepseek_ckpt_"))
    save_checkpoint(
        tempdir,
        model,
        step=0,
        config={
            "flavor": flavor,
            "seq_len": seq_len,
            "n_mtp_layers": n_mtp_layers,
            "index_topk": index_topk,
            "tie_word_embeddings": tie_word_embeddings,
            "swiglu_limit": swiglu_limit,
            "indexer_loss_coeff": indexer_loss_coeff,
        },
        tokenizer_hash=tokenizer.sha256,
    )
    reloaded = build_model(
        flavor,
        seq_len=seq_len,
        n_mtp_layers=n_mtp_layers,
        device=model.device,
        index_topk=index_topk,
        tie_word_embeddings=tie_word_embeddings,
        swiglu_limit=swiglu_limit,
        indexer_loss_coeff=indexer_loss_coeff,
    )
    load_checkpoint(tempdir, reloaded)
    batch = from_sequences([list(prompt) for prompt in prompts]).to(model.device)
    with torch.no_grad():
        original_logits = model.forward(batch, use_mtp=False)[0]
        reloaded_logits = reloaded.forward(batch, use_mtp=False)[0]
    diff = float((original_logits - reloaded_logits).abs().max())
    return CheckpointReport(
        max_abs_diff=diff,
        passed=diff == 0.0,
        num_tensors=len(model.state_dict()),
        checkpoint=str(tempdir),
    )


def run_all_checks(
    model: MiniDeepSeekModel,
    tokenizer: MiniDeepSeekTokenizer,
    prompts: Sequence[list[int]],
    *,
    flavor: str,
    seq_len: int,
    n_mtp_layers: int = 0,
    index_topk: int | None = None,
    tie_word_embeddings: bool | None = None,
    swiglu_limit: float | None = 10.0,
    indexer_loss_coeff: float = 0.01,
    max_new_tokens: int = 8,
    logprob_threshold: float = 1e-3,
    routing_threshold: float = 0.999,
) -> dict[str, Any]:
    logprob = check_logprob_consistency(
        model, tokenizer, prompts, max_new_tokens=max_new_tokens, threshold=logprob_threshold
    )
    routing = check_routing_consistency(
        model, tokenizer, prompts, max_new_tokens=max_new_tokens, threshold=routing_threshold
    )
    checkpoint = check_checkpoint_consistency(
        model,
        tokenizer,
        prompts,
        flavor=flavor,
        seq_len=seq_len,
        n_mtp_layers=n_mtp_layers,
        index_topk=index_topk,
        tie_word_embeddings=tie_word_embeddings,
        swiglu_limit=swiglu_limit,
        indexer_loss_coeff=indexer_loss_coeff,
    )
    return {
        "logprob": logprob.as_dict(),
        "routing": routing.as_dict(),
        "checkpoint": checkpoint.as_dict(),
        "passed": logprob.passed and routing.passed and checkpoint.passed,
    }
