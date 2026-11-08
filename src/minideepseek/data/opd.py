"""OPD prompt pool: domain-labelled prompts with a teacher key.

OPD is on-policy: the student samples its own trajectories, so the only data
needed is a prompt pool.  The ``teacher_key`` selects which teacher supplies the
target distribution; a hard key is used in v1 (soft weighting is a later
ablation, see roadmap section 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tokenizer import normalize_conversation
from .io import iter_jsonl, write_jsonl


@dataclass
class OPDPrompt:
    prompt: list[dict[str, Any]]
    teacher_key: str
    domain: str = "general"
    source: str = "unknown"
    multi_turn: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "teacher_key": self.teacher_key,
            "domain": self.domain,
            "source": self.source,
            **({"multi_turn": True} if self.multi_turn else {}),
        }


def sample_prompts(
    prompts: list[OPDPrompt],
    *,
    count: int,
    seed: int,
    step: int,
    lockstep: bool = False,
    teacher_weights: dict[str, float] | None = None,
) -> list[OPDPrompt]:
    """Draw one optimizer step's prompt batch deterministically.

    ``lockstep=True`` draws every prompt of the step from a single teacher
    (teachers weighted by ``teacher_weights``), which keeps the step
    domain-homogeneous.  ``False`` samples the mixed pool, the standard
    multi-teacher formulation.  Both are deterministic in ``seed + step``.
    """
    import random

    pool = list(prompts)
    if not pool:
        return []
    count = min(max(int(count), 1), len(pool))
    rng = random.Random(seed + step)
    if lockstep:
        by_teacher: dict[str, list[OPDPrompt]] = {}
        for prompt in pool:
            by_teacher.setdefault(prompt.teacher_key, []).append(prompt)
        keys = sorted(by_teacher)
        weights = [float((teacher_weights or {}).get(key, 1.0)) for key in keys]
        if sum(weights) <= 0:
            weights = [1.0] * len(keys)
        key = rng.choices(keys, weights=weights, k=1)[0]
        pool = by_teacher[key]
        count = min(count, len(pool))
    rng.shuffle(pool)
    return pool[:count]


def load_opd_prompts(path: str | Path) -> list[OPDPrompt]:
    prompts = []
    for item in iter_jsonl(path):
        prompt = item.get("prompt") or item.get("messages")
        if not prompt:
            continue
        try:
            prompt = normalize_conversation({"messages": prompt})
        except ValueError:
            continue
        key = item.get("teacher_key") or item.get("domain") or "general"
        prompts.append(
            OPDPrompt(
                prompt=prompt,
                teacher_key=str(key),
                domain=str(item.get("domain", key)),
                source=str(item.get("source", "unknown")),
                multi_turn=bool(
                    item.get("multi_turn")
                    or any(message.get("role") == "assistant" for message in prompt)
                ),
            )
        )
    return prompts


def build_opd_pool(
    inputs: list[str | Path],
    output_path: str | Path,
    *,
    teacher_keys: dict[str, str] | None = None,
    default_teacher: str = "general",
    max_per_source: int | None = None,
) -> dict[str, Any]:
    """Merge prompt sources into one mixed OPD pool.

    ``teacher_keys`` maps a source path (or its stem) to a teacher key.  Records
    that already carry ``teacher_key`` keep it.
    """
    teacher_keys = teacher_keys or {}
    merged: list[OPDPrompt] = []
    counts: dict[str, int] = {}
    for path in inputs:
        path = Path(path)
        key = teacher_keys.get(str(path)) or teacher_keys.get(path.stem) or default_teacher
        taken = 0
        for item in iter_jsonl(path):
            if max_per_source is not None and taken >= max_per_source:
                break
            prompt = item.get("prompt") or item.get("messages") or item.get("conversations")
            if not prompt:
                continue
            prompt = [
                message for message in prompt if isinstance(message, dict)
            ]
            # Strip only the trailing assistant turn (the student generates
            # it); earlier assistant/tool turns are legitimate multi-turn
            # context for agent prompts and must be kept.
            while prompt and prompt[-1].get("role") == "assistant":
                prompt = prompt[:-1]
            if not prompt:
                continue
            multi_turn = bool(
                item.get("multi_turn")
                or any(message.get("role") == "assistant" for message in prompt)
            )
            merged.append(
                OPDPrompt(
                    prompt=prompt,
                    teacher_key=str(item.get("teacher_key") or key),
                    domain=str(item.get("domain") or key),
                    source=path.stem,
                    multi_turn=multi_turn,
                )
            )
            taken += 1
            counts[path.stem] = counts.get(path.stem, 0) + 1
    if not merged:
        raise ValueError("no OPD prompts were produced")
    write_jsonl(output_path, (prompt.to_dict() for prompt in merged))
    return {
        "format": "minideepseek-opd-v1",
        "prompts": len(merged),
        "per_source": counts,
        "teacher_keys": teacher_keys,
    }


def balance_by_teacher(prompts: list[OPDPrompt], seed: int = 1234) -> list[OPDPrompt]:
    """Interleave teacher keys so every batch is roughly 1:1:1."""
    import random

    by_key: dict[str, list[OPDPrompt]] = {}
    for prompt in prompts:
        by_key.setdefault(prompt.teacher_key, []).append(prompt)
    rng = random.Random(seed)
    for values in by_key.values():
        rng.shuffle(values)
    keys = sorted(by_key)
    interleaved: list[OPDPrompt] = []
    index = 0
    while True:
        added = False
        for key in keys:
            if index < len(by_key[key]):
                interleaved.append(by_key[key][index])
                added = True
        if not added:
            break
        index += 1
    return interleaved
