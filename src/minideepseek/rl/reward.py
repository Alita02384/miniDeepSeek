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
from .tools import parse_tool_calls, valid_tool_call

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_RE = re.compile(
    r"(?i)(?:final\s+answer|answer\s*[:：]|答案是|答案为)\s*[:：]? *([^\n]{1,120})"
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")
#: ``x = 42`` is an explicit answer even without "answer:"; bare prose numbers
#: are not (see the no-fallback rule below).
_ASSIGNMENT_RE = re.compile(
    r"(?<![\w.])([A-Za-z]\w*)\s*=\s*(-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?)"
)


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
        # last numeric token when one is present.  Strip thousands separators
        # first so "1,234" is one number, not "234".
        numbers = _NUMBER_RE.findall(captured.replace(",", ""))
        if numbers:
            return numbers[-1].strip()
        return captured
    match = _ASSIGNMENT_RE.search(text)
    if match:
        return match.group(2).strip()
    # No explicit answer marker: do not guess from the last number in prose.
    # "the numbers are 3 and 5" must not be scored as the reference answer.
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
    """Unit-test reward with configurable pass/compile/format weights.

    The tests live in ``record.rubric['tests']`` (or ``test_code``).  The
    settings are deliberately explicit so the code specialist's reward can be
    tuned later without editing the executor.  Every component is already
    weighted, so ``sum(components.values()) == total`` for diagnostics.
    """
    code = _extract_code(response)
    tests = record.rubric.get("tests") or record.rubric.get("test_code")
    if not code.strip():
        return RewardResult(
            total=0.0,
            components={"unit_tests": 0.0, "compiled": 0.0, "format": 0.0},
            details={"error": "empty code"},
        )
    format_bonus = float(settings.code_format_bonus) if code != response else 0.0
    if not tests:
        return RewardResult(
            total=0.0,
            components={"unit_tests": 0.0, "compiled": 0.0, "format": 0.0},
            details={"error": "record has no rubric.tests"},
        )
    try:
        ast.parse(code)
    except SyntaxError as error:
        penalty = -float(settings.code_syntax_penalty)
        return RewardResult(
            total=penalty,
            components={"unit_tests": 0.0, "compiled": 0.0, "format": 0.0, "syntax": penalty},
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
            components = {
                "unit_tests": 0.0,
                "compiled": float(settings.code_compile_bonus),
                "format": format_bonus,
            }
            return RewardResult(
                total=sum(components.values()),
                components=components,
                details={"error": "timeout"},
            )
    passed = completed.returncode == 0
    components = {
        "unit_tests": float(settings.code_pass_reward) if passed else 0.0,
        "compiled": float(settings.code_compile_bonus),
        "format": format_bonus,
    }
    return RewardResult(
        total=sum(components.values()),
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




def _length_shaping(text: str, settings: RewardSettings) -> float:
    if settings.length_bonus <= 0:
        return 0.0
    return settings.length_bonus if 20 <= len(text.strip()) <= 800 else -settings.length_bonus


def _think_shaping(text: str, settings: RewardSettings) -> float:
    if settings.think_bonus <= 0 or "</think>" not in text:
        return 0.0
    thinking, _answer = text.split("</think>", 1)
    value = settings.think_bonus if 20 <= len(thinking.strip()) <= 300 else -settings.think_bonus
    value += 0.25 if text.count("</think>") == 1 else -0.25
    return value


def _repetition_shaping(text: str, settings: RewardSettings) -> float:
    cap = float(settings.repetition_penalty)
    if cap <= 0:
        return 0.0
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    if not grams:
        return 0.0
    repeats = len(grams) - len(set(grams))
    return min(cap, repeats * cap * 2 / len(grams))


def _finalize(
    result: RewardResult,
    response: str,
    settings: RewardSettings,
    *,
    apply_shaping: bool = True,
) -> RewardResult:
    """Apply MiniMind-style dense shaping and optional reward clipping.

    ``reward_clip`` bounds the base reward (the reward model / verifier), not
    the shaping terms.  Clipping the sum made every sufficiently good group
    collapse to the same value -- ``group_advantages`` then returned all zeros
    and the group silently stopped learning.  Shaping is deliberately kept
    outside the clip so a good group still has intra-group differences.
    """
    shaped = dict(result.components)
    details = dict(result.details)
    base_total = float(result.total)
    if settings.reward_clip > 0:
        clipped = max(-settings.reward_clip, min(settings.reward_clip, base_total))
        if clipped != base_total:
            details["pre_clip_total"] = base_total
            shaped["clip"] = clipped - base_total
            base_total = clipped

    shaping: dict[str, float] = {}
    if apply_shaping:
        if settings.length_bonus > 0:
            shaping["length"] = _length_shaping(response, settings)
        if settings.think_bonus > 0:
            shaping["think"] = _think_shaping(response, settings)
        if settings.repetition_penalty > 0:
            shaping["repetition"] = -_repetition_shaping(response, settings)
    # A reward may already expose a component with the shaping name; keep its
    # own value rather than silently overwriting it.
    for key, value in shaping.items():
        if key not in shaped:
            shaped[key] = value
    total = base_total + sum(shaping.values())
    return RewardResult(total=total, components=shaped, details=details)


def _gt_hits(text: str, gt: list[str]) -> set[str]:
    """Match MiniMind's final-answer check: exact substring or numeric equality."""
    text_value = str(text)
    text_number = text_value.replace(",", "")
    numbers = [float(value) for value in re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])", text_number)]
    hits: set[str] = set()
    for value in gt:
        candidate = str(value).strip()
        if not candidate:
            continue
        if candidate.lower() in text_value.lower():
            hits.add(candidate)
            continue
        normalized = candidate.replace(",", "")
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized):
            target = float(normalized)
            if any(abs(target - number) < 1e-6 for number in numbers):
                hits.add(candidate)
    return hits


def general_reward(
    response: str,
    record: GRPORecord,
    settings: RewardSettings,
    *,
    reward_model=None,
) -> RewardResult:
    """MiniMind RLAIF path: dense reward model + format shaping.

    The MiniMind prompt pool (``rlaif.jsonl``) has no verifiable answer, so the
    reward model is the primary signal.  Without one we still return the format
    shaping so a smoke run does not silently train on a zero reward.
    """
    score = 0.0
    # MiniMind feeds the answer after the think block to the reward model, not
    # the raw reasoning trace (which the RM was not trained to score).
    answer = response.split("</think>")[-1] if "</think>" in response else response
    if reward_model is not None:
        score = float(reward_model.get_score(record.prompt, answer))
    return RewardResult(
        total=score,
        components={"reward_model": score},
        details={"reward_model": reward_model is not None, "answer": answer},
    )


def tool_reward(
    response: str,
    record: GRPORecord,
    settings: RewardSettings,
    *,
    reward_model=None,
) -> RewardResult:
    """MiniMind ``train_agent.py`` tool/agent reward.

    ``record.rubric['tool_turns']`` is filled by the multi-turn rollout engine;
    ``record.rubric['unfinished']`` marks a trajectory that hit ``max_turns``
    or ran out of context.  An unfinished trajectory gets no gt credit: the
    final answer came from a turn that never saw the tool result.
    """
    turns = record.rubric.get("tool_turns") or [response]
    turns = [str(turn) for turn in turns]
    calls = [call for turn in turns for call in parse_tool_calls(turn)]
    tools = record.resolved_tools()
    tag_balance = sum(
        abs(turn.count("<tool_call>") - turn.count("</tool_call>")) for turn in turns
    )
    total = -float(settings.tool_tag_penalty) * tag_balance
    components = {"tool_tag_balance": -float(settings.tool_tag_penalty) * tag_balance}
    unfinished = bool(record.rubric.get("unfinished", False))
    details: dict[str, Any] = {
        "turns": len(turns),
        "tool_calls": len(calls),
        "has_calls": bool(calls),
        "unfinished": unfinished,
    }

    if not calls:
        # No tool call: fall back to the ordinary answer branch.  Dense
        # length/think shaping still applies in this branch (MiniMind only
        # applies it when there is no tool call).
        if reward_model is not None and not unfinished:
            answer = response.split("</think>")[-1] if "</think>" in response else response
            rm_score = float(reward_model.get_score(record.prompt, answer))
            total += rm_score
            components["reward_model"] = rm_score
        if unfinished:
            penalty = -float(settings.tool_unfinished_penalty)
            total += penalty
            components["unfinished"] = penalty
        return RewardResult(total=total, components=components, details=details)

    gt = list(record.gt)
    if not gt:
        rubric_gt = record.rubric.get("gt")
        if isinstance(rubric_gt, (list, tuple)):
            gt = [str(value) for value in rubric_gt]
        elif record.reference_answer:
            gt = [record.reference_answer]
    valid_calls = sum(1 for call in calls if valid_tool_call(call, tools))
    expected = record.rubric.get("expected_calls")
    if expected is None:
        # Do not use ``len(gt)`` as the expected call count: one weather call
        # can return several gt values, and the old code then penalised a
        # correct single call by ``abs(1 - len(gt))``.
        gap = max(0, len(calls) - valid_calls)
    else:
        gap = abs(valid_calls - int(expected)) + max(0, len(calls) - valid_calls)
    align = float(settings.tool_valid_bonus) if gap == 0 else -float(settings.tool_valid_bonus) * gap
    total += align
    components["tool_alignment"] = align
    if unfinished:
        # No gt credit: the final answer was produced without seeing this
        # turn's observation (or the turn was cut off mid-generation).
        penalty = -float(settings.tool_unfinished_penalty)
        total += penalty
        components["unfinished"] = penalty
        details.update({"valid_calls": valid_calls, "gt": gt, "hits": []})
        return RewardResult(total=total, components=components, details=details)
    final_turn = turns[-1] if turns else response
    final_text = final_turn.split("</think>")[-1] if "</think>" in final_turn else final_turn
    hits = _gt_hits(final_text, gt) if gt else set()
    gt_score = float(settings.tool_gt_weight) * (len(hits) / len(gt)) if gt else 0.0
    total += gt_score
    components["gt_match"] = gt_score
    details.update({"valid_calls": valid_calls, "gt": gt, "hits": sorted(hits)})
    return RewardResult(total=total, components=components, details=details)


def dispatch_reward(
    response: str,
    record: GRPORecord,
    settings: RewardSettings,
    *,
    judge: Judge | None = None,
    reward_model=None,
) -> RewardResult:
    is_tool = record.verifier == "tool" or settings.kind == "tool"
    if is_tool:
        result = tool_reward(response, record, settings, reward_model=reward_model)
    elif record.verifier == "math_exact_match" or settings.kind == "math":
        result = math_reward(response, record, settings)
    elif record.verifier == "unit_test" or settings.kind == "code":
        result = code_reward(response, record, settings)
    elif record.domain == "medical" or settings.kind == "medical":
        result = medical_reward(response, record, settings)
    elif record.verifier == "llm_judge" or settings.kind == "world":
        result = world_reward(response, record, settings, judge=judge)
    elif settings.kind in {"general", "rlaif"}:
        result = general_reward(response, record, settings, reward_model=reward_model)
    else:
        # Unknown verifier: no learning signal rather than a silently wrong one.
        result = RewardResult(total=0.0, components={}, details={"verifier": record.verifier})
    # Tool trajectories are scored by tool alignment + gt; MiniMind does not
    # add length/think shaping once the model actually called a tool.  An
    # unfinished trajectory gets no shaping either.
    apply_shaping = not (
        is_tool and (result.details.get("has_calls") or result.details.get("unfinished"))
    )
    return _finalize(result, response, settings, apply_shaping=apply_shaping)


class _LazyRewardModel:
    """Load the 1.8B reward model on first ``get_score``.

    ``tool`` only consults the RM when the model made no tool call; loading it
    eagerly at startup costs 3.6 GB and seconds even for runs that never hit
    that branch.  General hits it on the first sample.
    """

    def __init__(self, path: str, dtype: str):
        self.path = path
        self.dtype = dtype
        self._model = None

    def _load(self):
        if self._model is None:
            from .reward_model import LocalRewardModel

            self._model = LocalRewardModel(self.path, dtype=self.dtype)
        return self._model

    def get_score(self, messages, response):
        return self._load().get_score(messages, response)


def make_reward_fn(
    settings: RewardSettings,
    *,
    judge: Judge | None = None,
    reward_model=None,
) -> Callable[[str, GRPORecord], RewardResult]:
    if judge is None and settings.kind == "world" and settings.world_judge == "http":
        from .judge import make_judge

        judge = make_judge(settings)
    if (
        reward_model is None
        and settings.kind in {"general", "rlaif"}
        and not settings.reward_model_path
    ):
        raise ValueError(
            "general reward needs a reward model: set reward.reward_model_path "
            "or pass one explicitly (a zero reward would silently train the "
            "specialist on shaping only)"
        )
    if reward_model is None and settings.reward_model_path:
        reward_model = _LazyRewardModel(
            settings.reward_model_path,
            dtype=settings.reward_model_dtype,
        )

    def reward_fn(response: str, record: GRPORecord) -> RewardResult:
        return dispatch_reward(
            response, record, settings, judge=judge, reward_model=reward_model
        )

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
