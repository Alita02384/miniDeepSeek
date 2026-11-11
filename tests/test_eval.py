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


def test_eval_code_maps_top_level_tests_and_scores_pass_rate(tmp_path, stub_tokenizer):
    from minideepseek.data.grpo import load_records
    from minideepseek.rl.reward import RewardResult

    path = tmp_path / "code.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "write add"}],
                "tests": "assert add(1, 2) == 3",
                "domain": "code",
            }
        )
        + "\n"
    )
    records = load_records(path)
    assert records[0].verifier == "unit_test"
    assert records[0].rubric["tests"] == "assert add(1, 2) == 3"

    seen = []

    def reward_fn(response, record):
        seen.append((record.verifier, record.rubric.get("tests")))
        return RewardResult(total=1.0, components={"unit_tests": 1.0})

    config = _eval_config(
        tmp_path,
        domain="code",
        data_path=str(path),
        limit=1,
        reward=RewardSettings(kind="code"),
    )
    report = evaluate(config, tokenizer=stub_tokenizer, reward_fn=reward_fn)
    assert seen == [("unit_test", "assert add(1, 2) == 3")]
    assert report.metrics["pass_rate"] == 1.0
    assert report.details["reward_source"] == "injected"


def test_eval_general_uses_injected_reward_and_honours_thinking_ratio(tmp_path, stub_tokenizer):
    from minideepseek.rl.reward import RewardResult

    path = tmp_path / "general.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "hello"}],
                "domain": "general",
            }
        )
        + "\n"
    )
    config = _eval_config(
        tmp_path,
        domain="general",
        data_path=str(path),
        limit=1,
        thinking_ratio=0.0,
        reward=RewardSettings(kind="general"),
    )
    report = evaluate(
        config,
        tokenizer=stub_tokenizer,
        reward_fn=lambda response, record: RewardResult(
            total=1.5, components={"reward_model": 1.5}
        ),
    )
    assert report.metrics["reward_model_mean"] == 1.5
    assert report.details["reward_kind"] == "general"
    assert report.details["thinking_rate"] == 0.0


def test_eval_tool_dispatches_to_tool_reward_not_world(tmp_path, stub_tokenizer):
    path = tmp_path / "tool.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "1+1"}],
                "domain": "tool",
            }
        )
        + "\n"
    )
    config = _eval_config(
        tmp_path,
        domain="tool",
        data_path=str(path),
        limit=1,
        thinking_ratio=0.1,
        reward=RewardSettings(kind="tool"),
    )
    report = evaluate(config, tokenizer=stub_tokenizer)
    assert report.details["reward_kind"] == "tool"
    assert report.details["tool_execution"] is False
    assert "tool_reward_mean" in report.metrics
    assert "rubric_factuality_mean" not in report.metrics
