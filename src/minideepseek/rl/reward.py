"""Reward functions for GRPO and evaluation.

Rewards are deliberately decomposed into named components so a run can tell
"the model got more correct" apart from "the model got longer / more fluent".
The world-domain path accepts an offline judge callable and keeps the full
multi-dimensional rubric; the default heuristic is only a placeholder for tests
and smoke runs, not a substitute for an API judge or a trained GRM.
"""

from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import RewardSettings
from ..data.grpo import GRPORecord

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_RE = re.compile(
    r"(?i)(?:final\s+answer|answer\s*[:：]|答案是|答案为)\s*[:：]? *([^\n]{1,120})"
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")


@dataclass
class RewardResult:
    total: float
    components: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"total": self.total, "components": self.components, "details": self.details}


class Judge(Protocol):
    def __call__(self, response: str, record: GRPORecord) -> dict[str, float]: ...


def extract_answer(text: str) -> str | None:
    match = _BOXED_RE.search(text)
    if match:
        return match.group(1).strip()
    match = _ANSWER_RE.search(text)
    if match:
        captured = match.group(1).strip()
        # The capture may include units/words ("42 元", "x = 42"); prefer the
        # last numeric token when one is present.
        numbers = _NUMBER_RE.findall(captured)
        if numbers:
            return numbers[-1].strip()
        return captured
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].strip()
    return None


def normalize_answer(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    value = value.replace("$", "").replace(",", "").replace(" ", "")
    value = value.rstrip("。.;；")
    return value


def _to_number(value: str) -> float | None:
    value = normalize_answer(value)
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:/(-?\d+(?:\.\d+)?))?", value)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2)) if match.group(2) else 1.0
        return numerator / denominator if denominator else None
    try:
        return float(value)
    except ValueError:
        return None


def math_correct(response: str, reference: str | None, *, tolerance: float = 1e-4) -> tuple[bool, str | None]:
    if not reference:
        return False, None
    extracted = extract_answer(response)
    if extracted is None:
        return False, None
    if normalize_answer(extracted) == normalize_answer(reference):
        return True, extracted
    left = _to_number(extracted)
    right = _to_number(reference)
    if left is not None and right is not None:
        return abs(left - right) <= tolerance * max(1.0, abs(right)), extracted
    return False, extracted


def length_penalty(text: str, settings: RewardSettings) -> float:
    if settings.length_penalty <= 0:
        return 0.0
    limit = max(int(settings.max_length_penalty_at), 1)
    # Characters are a cheap, tokenizer-independent proxy for response length.
    excess = max(len(text) - limit, 0) / limit
    return min(1.0, excess) * settings.length_penalty


def math_reward(response: str, record: GRPORecord, settings: RewardSettings) -> RewardResult:
    correct, extracted = math_correct(response, record.reference_answer)
    # Rollout opens the think block in the prompt; a proper response closes it.
    has_think = "</think>" in response
    components = {
        "correctness": 1.0 if correct else 0.0,
        "format": settings.format_bonus if has_think else 0.0,
        "parse": settings.parse_bonus if extracted is not None else 0.0,
        "length_penalty": -length_penalty(response, settings),
    }
    return RewardResult(
        total=sum(components.values()),
        components=components,
        details={"extracted": extracted, "reference": record.reference_answer},
    )


def _extract_code(response: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", response, flags=re.S)
    if blocks:
        return "\n".join(blocks)
    return response


def code_reward(response: str, record: GRPORecord, settings: RewardSettings) -> RewardResult:
    code = _extract_code(response)
    tests = record.rubric.get("tests") or record.rubric.get("test_code")
    if not tests:
        return RewardResult(
            total=0.0,
            components={"unit_tests": 0.0, "compiled": 0.0},
            details={"error": "record has no rubric.tests"},
        )
    try:
        ast.parse(code)
        compiled = 1.0
    except SyntaxError as error:
        return RewardResult(
            total=-0.1,
            components={"unit_tests": 0.0, "compiled": 0.0},
            details={"error": f"syntax: {error}"},
        )
    program = code + "\n\n" + str(tests) + "\n"
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "candidate.py"
        script.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True,
                text=True,
                timeout=float(settings.code_timeout_s),
                cwd=directory,
            )
        except subprocess.TimeoutExpired:
            return RewardResult(
                total=0.0,
                components={"unit_tests": 0.0, "compiled": compiled},
                details={"error": "timeout"},
            )
    passed = completed.returncode == 0
    components = {"unit_tests": 1.0 if passed else 0.0, "compiled": compiled}
    return RewardResult(
        total=1.0 if passed else 0.0,
        components=components,
        details={
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        },
    )


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", a.lower()))
    tokens_b = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", b.lower()))
    if not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_b)


def default_world_judge(response: str, record: GRPORecord) -> dict[str, float]:
    reference = record.reference_answer or ""
    factuality = _token_overlap(response, reference) if reference else 0.0
    prompt_text = " ".join(str(message.get("content", "")) for message in record.prompt)
    relevance = _token_overlap(response, prompt_text)
    length_score = min(len(response) / 400.0, 1.0)
    sentences = max(1, len(re.findall(r"[。！？.!?]", response)))
    clarity = min(1.0, sentences / 5.0)
    fluency = 1.0 if response.strip() else 0.0
    return {
        "factuality": factuality,
        "relevance": relevance,
        "completeness": length_score,
        "clarity": clarity,
        "fluency": fluency,
        "citation_support": 0.0,
    }


def world_reward(
    response: str,
    record: GRPORecord,
    settings: RewardSettings,
    *,
    judge: Judge | None = None,
) -> RewardResult:
    scores = (judge or default_world_judge)(response, record)
    for key in settings.world_weights:
        scores.setdefault(key, 0.0)
    for key in scores:
        scores[key] = max(0.0, min(1.0, float(scores[key])))
    total = sum(float(settings.world_weights.get(key, 0.0)) * value for key, value in scores.items())
    total -= length_penalty(response, settings)
    return RewardResult(total=total, components=dict(scores), details={"rubric": scores})


def medical_reward(response: str, record: GRPORecord, settings: RewardSettings) -> RewardResult:
    safety_cues = ("就医", "医生", "医院", "急诊", "专业", "consult", "doctor", "hospital")
    unsafe_claims = ("一定是", "百分百", "绝对", "无需就医", "guaranteed")
    safety = 1.0 if any(cue in response for cue in safety_cues) else 0.0
    if any(claim in response for claim in unsafe_claims):
        safety -= 0.5
    factual = _token_overlap(response, record.reference_answer or "")
    components = {"safety": max(0.0, safety), "factuality": factual}
    total = 0.6 * components["safety"] + 0.4 * factual - length_penalty(response, settings)
    return RewardResult(total=total, components=components, details={})


def dispatch_reward(
    response: str,
    record: GRPORecord,
    settings: RewardSettings,
    *,
    judge: Judge | None = None,
) -> RewardResult:
    if record.verifier == "math_exact_match" or settings.kind == "math":
        return math_reward(response, record, settings)
    if record.verifier == "unit_test" or settings.kind == "code":
        return code_reward(response, record, settings)
    if record.domain == "medical" or settings.kind == "medical":
        return medical_reward(response, record, settings)
    if record.verifier == "llm_judge" or settings.kind == "world":
        return world_reward(response, record, settings, judge=judge)
    # Unknown verifier: no learning signal rather than a silently wrong one.
    return RewardResult(total=0.0, components={}, details={"verifier": record.verifier})


def make_reward_fn(settings: RewardSettings, *, judge: Judge | None = None) -> Callable[[str, GRPORecord], RewardResult]:
    if judge is None and settings.kind == "world" and settings.world_judge == "http":
        from .judge import make_judge

        judge = make_judge(settings)

    def reward_fn(response: str, record: GRPORecord) -> RewardResult:
        return dispatch_reward(response, record, settings, judge=judge)

    return reward_fn


def group_advantages(rewards: list[float], *, eps: float = 1e-6) -> list[float]:
    """Zero-mean, unit-variance advantages within one prompt group."""
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std < eps:
        return [0.0 for _ in rewards]
    return [(reward - mean) / (std + eps) for reward in rewards]
