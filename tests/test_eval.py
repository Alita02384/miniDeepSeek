from __future__ import annotations

import json

from minideepseek.config import EvalConfig, ModelSettings, RewardSettings
from minideepseek.eval import evaluate, write_report


def _eval_config(tmp_path, domain="math", **overrides) -> EvalConfig:
    path = tmp_path / "eval.jsonl"
    rows = [
        {
            "prompt": [{"role": "user", "content": "What is 1+1?"}],
            "reference_answer": "2",
            "domain": "math",
            "verifier": "math_exact_match",
        },
        {
            "prompt": [{"role": "user", "content": "What is 2+2?"}],
            "reference_answer": "4",
            "domain": "math",
            "verifier": "math_exact_match",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    kwargs = dict(
        model=ModelSettings(
            flavor="debugmodel",
            seq_len=128,
            n_mtp_layers=0,
            swiglu_limit=10.0,
            device="cpu",
            dtype="float32",
            seed=0,
        ),
        checkpoint=None,
        data_path=str(path),
        domain=domain,
        max_new_tokens=4,
        temperature=0.0,
        limit=2,
        out_path=str(tmp_path / "report.json"),
        reward=RewardSettings(kind=domain),
    )
    kwargs.update(overrides)
    return EvalConfig(**kwargs)


def test_evaluate_math_with_stub_tokenizer(tmp_path, stub_tokenizer):
    report = evaluate(_eval_config(tmp_path), tokenizer=stub_tokenizer)
    assert report.samples == 2
    assert "reward_mean" in report.metrics
    assert "accuracy" in report.metrics
    assert report.details["tokenizer_hash"] == stub_tokenizer.sha256
    path = write_report(report, tmp_path / "report.json")
    assert path.exists()
    assert path.with_suffix(".md").exists()
    payload = json.loads(path.read_text())
    assert payload["domain"] == "math"
    assert len(payload["examples"]) == 2


def test_evaluate_world_with_judge(tmp_path, stub_tokenizer):
    path = tmp_path / "world.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "Capital of France?"}],
                "reference_answer": "Paris",
                "domain": "world",
                "verifier": "llm_judge",
            }
        )
        + "\n"
    )
    config = _eval_config(tmp_path, domain="world", data_path=str(path), limit=1)
    report = evaluate(
        config,
        tokenizer=stub_tokenizer,
        judge=lambda response, record: {
            "factuality": 1.0,
            "relevance": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "fluency": 1.0,
            "citation_support": 0.0,
        },
    )
    assert report.samples == 1
    assert "rubric_factuality_mean" in report.metrics
    assert report.metrics["rubric_factuality_mean"] == 1.0
