"""World-domain judges: heuristic default and an optional HTTP judge.

The heuristic in ``reward.default_world_judge`` is a placeholder.  Production
world GRPO should use an API judge or a local generative reward model.  This
module implements the API path with a strict JSON contract; it is optional and
only activates when ``RewardSettings.world_judge == "http"``.
"""

from __future__ import annotations

import json
import os
import urllib.request

from ..config import RewardSettings
from ..data.grpo import GRPORecord
from .reward import default_world_judge

DIMENSIONS = (
    "factuality",
    "relevance",
    "completeness",
    "clarity",
    "fluency",
    "citation_support",
)

_PROMPT = """You are grading a world-knowledge answer. Return JSON only:
{{"factuality":0-1,"relevance":0-1,"completeness":0-1,"clarity":0-1,"fluency":0-1,"citation_support":0-1}}

Question:
{prompt}

Reference (may be empty):
{reference}

Answer:
{answer}
"""


def build_judge_prompt(response: str, record: GRPORecord) -> str:
    prompt = " ".join(
        str(message.get("content", "")) for message in record.prompt
    )
    return _PROMPT.format(
        prompt=prompt[-4000:],
        reference=(record.reference_answer or "")[:4000],
        answer=response[:8000],
    )


class HTTPJudge:
    """OpenAI-compatible chat-completions judge with JSON output."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        dimensions: tuple[str, ...] = DIMENSIONS,
    ):
        self.url = url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url = self.url + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.dimensions = dimensions

    def _request(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {self.api_key}"}
                    if self.api_key
                    else {}
                ),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode())
        return body["choices"][0]["message"]["content"]

    def __call__(self, response: str, record: GRPORecord) -> dict[str, float]:
        content = self._request(build_judge_prompt(response, record))
        scores = json.loads(content)
        return {
            dimension: max(0.0, min(1.0, float(scores.get(dimension, 0.0))))
            for dimension in self.dimensions
        }


def make_judge(settings: RewardSettings):
    """Return the configured world judge, falling back to the heuristic."""
    if settings.world_judge != "http":
        return default_world_judge
    url = os.environ.get("MINIDEEPSEEK_JUDGE_URL") or os.environ.get(
        "MINIDEEPSEEK_JUDGE_URL"
    )
    model = os.environ.get("MINIDEEPSEEK_JUDGE_MODEL")
    if not url or not model:
        raise RuntimeError(
            "world_judge=http requires MINIDEEPSEEK_JUDGE_URL and "
            "MINIDEEPSEEK_JUDGE_MODEL"
        )
    return HTTPJudge(
        url=url,
        model=model,
        api_key=os.environ.get("MINIDEEPSEEK_JUDGE_API_KEY"),
        timeout_s=settings.world_judge_timeout_s,
    )
