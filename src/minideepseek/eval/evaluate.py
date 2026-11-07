"""Evaluation runners for math, code, and rubric-scored world tasks.

All three use the same generation path as GRPO.  Metrics are intentionally
separated from training rewards: evaluation never updates weights and records
format/parse failures alongside accuracy.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import EvalConfig, resolve_device
from ..data.grpo import GRPORecord, load_records
from ..model.checkpoint import load_checkpoint, read_metadata
from ..model.wrapper import MiniDeepSeekModel, build_model
from ..rl.reward import RewardResult, code_reward, math_reward, world_reward
from ..tokenizer import MiniDeepSeekTokenizer, load_tokenizer


@dataclass
class EvalSample:
    prompt: str
    response: str
    reference: str | None
    reward: RewardResult


@dataclass
class EvalReport:
    domain: str
    samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "samples": self.samples,
            "metrics": self.metrics,
            "details": self.details,
            "examples": self.examples,
        }


def _load_model(config: EvalConfig, tokenizer: MiniDeepSeekTokenizer) -> MiniDeepSeekModel:
    device = resolve_device(config.model.device)
    runtime_topk = None
    if config.checkpoint:
        runtime_topk = read_metadata(config.checkpoint).get("extra", {}).get("index_topk")
    settings = config.model
    if runtime_topk is not None:
        settings = dataclasses.replace(settings, index_topk=int(runtime_topk))
    model = build_model(settings, device=device, dtype=settings.dtype)
    if config.checkpoint:
        meta = load_checkpoint(
            config.checkpoint,
            model,
            map_location="cpu",
            ignore_prefixes=("mtp_layers.",),
        )
        expected = meta.get("tokenizer_hash")
        if expected and expected != tokenizer.sha256:
            raise ValueError(
                f"checkpoint tokenizer hash mismatch: {expected} != {tokenizer.sha256}"
            )
    model.eval()
    return model


def _generate_responses(
    model: MiniDeepSeekModel,
    tokenizer: MiniDeepSeekTokenizer,
    records: Sequence[GRPORecord],
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> list[str]:
    prompts = [
        tokenizer.encode(
            tokenizer.render_chat(
                record.prompt,
                add_generation_prompt=True,
                tools=record.resolved_tools(),
                open_thinking=True,
            )
        )
        for record in records
    ]
    if temperature <= 0:
        output = model.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            greedy=True,
            stop_token_ids=tokenizer.stop_token_ids,
            seed=seed,
        )
    else:
        output = model.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_token_ids=tokenizer.stop_token_ids,
            seed=seed,
        )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output.response_ids]


def evaluate(
    config: EvalConfig,
    *,
    judge: Callable[[str, GRPORecord], dict[str, float]] | None = None,
    tokenizer: MiniDeepSeekTokenizer | None = None,
) -> EvalReport:
    if not config.data_path:
        raise ValueError("EvalConfig.data_path is required")
    tokenizer = tokenizer or load_tokenizer(vocab_limit=config.model.vocab_limit)
    model = _load_model(config, tokenizer)
    records = load_records(config.data_path)
    if config.limit is not None:
        records = records[: config.limit]
    responses = _generate_responses(
        model,
        tokenizer,
        records,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        seed=config.model.seed,
    )
    settings = config.reward
    if judge is None and settings.world_judge == "http":
        from ..rl.judge import make_judge

        judge = make_judge(settings)
    samples: list[EvalSample] = []
    for record, response in zip(records, responses, strict=True):
        if config.domain == "math" or record.verifier == "math_exact_match":
            result = math_reward(response, record, settings)
        elif config.domain == "code" or record.verifier == "unit_test":
            result = code_reward(response, record, settings)
        else:
            result = world_reward(response, record, settings, judge=judge)
        samples.append(
            EvalSample(
                prompt=record.prompt[-1].get("content", "") if record.prompt else "",
                response=response,
                reference=record.reference_answer,
                reward=result,
            )
        )
    report = _summarize(config.domain, samples)
    report.details = {
        "checkpoint": config.checkpoint,
        "flavor": config.model.flavor,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "tokenizer_hash": tokenizer.sha256,
    }
    return report


def _summarize(domain: str, samples: list[EvalSample]) -> EvalReport:
    if not samples:
        return EvalReport(domain=domain, samples=0)
    totals = [sample.reward.total for sample in samples]
    accuracy = sum(1.0 for sample in samples if sample.reward.components.get("correctness", 0.0) > 0) / len(samples)
    parse = sum(1.0 for sample in samples if sample.reward.components.get("parse", 0.0) > 0) / len(samples)
    format_ok = sum(1.0 for sample in samples if sample.reward.components.get("format", 0.0) > 0) / len(samples)
    unit = sum(1.0 for sample in samples if sample.reward.components.get("unit_tests", 0.0) > 0) / len(samples)
    metrics = {
        "reward_mean": statistics.fmean(totals),
        "reward_std": statistics.pstdev(totals) if len(totals) > 1 else 0.0,
        "response_chars_mean": statistics.fmean(len(sample.response) for sample in samples),
    }
    if domain == "math":
        metrics["accuracy"] = accuracy
        metrics["parse_rate"] = parse
        metrics["format_rate"] = format_ok
    elif domain == "code":
        metrics["pass_rate"] = unit
    else:
        component_names = sorted(
            {key for sample in samples for key in sample.reward.components}
        )
        for name in component_names:
            metrics[f"rubric_{name}_mean"] = statistics.fmean(
                sample.reward.components.get(name, 0.0) for sample in samples
            )
    examples = [
        {
            "prompt": sample.prompt[:500],
            "reference": sample.reference,
            "response": sample.response[:1000],
            "reward": sample.reward.as_dict(),
        }
        for sample in samples[:10]
    ]
    return EvalReport(domain=domain, samples=len(samples), metrics=metrics, examples=examples)


def write_report(report: EvalReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown_path = path.with_suffix(".md")
    lines = [
        f"# Evaluation report: {report.domain}",
        "",
        f"- samples: {report.samples}",
        *[f"- {key}: {value:.4f}" for key, value in report.metrics.items()],
        "",
        "## Examples",
        "",
    ]
    for example in report.examples[:5]:
        lines.extend(
            [
                "### Prompt",
                "```",
                example["prompt"],
                "```",
                "### Response",
                "```",
                example["response"],
                "```",
                f"reward: `{json.dumps(example['reward'], ensure_ascii=False)}`",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return path
