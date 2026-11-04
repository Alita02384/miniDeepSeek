from __future__ import annotations

import importlib.util

import pytest

from minideepseek.config import RLConfig
from minideepseek.data.grpo import GRPORecord
from minideepseek.rl.backends import build_rollout_engine
from minideepseek.rl.rollout import TorchRolloutEngine
from minideepseek.rl.verl import (
    build_verl_config,
    build_verl_opd_config,
    compute_score,
    flatten_overrides,
    to_verl_record,
)


def test_verl_record_schema_and_reward(debug_model, stub_tokenizer):
    record = GRPORecord(
        prompt=[{"role": "user", "content": "1+1?"}],
        domain="math",
        reference_answer="2",
        verifier="math_exact_match",
    )
    row = to_verl_record(record)
    assert row["data_source"] == "minideepseek/math/math_exact_match"
    assert row["reward_model"]["ground_truth"] == "2"
    assert row["extra_info"]["verifier"] == "math_exact_match"
    assert compute_score(row["data_source"], "答案 2", "2", row["extra_info"]) > 0


def test_verl_config_contains_hf_and_reward(tmp_path):
    config = build_verl_config(
        hf_model_path=tmp_path / "hf",
        train_files=["train.parquet"],
        val_files=["val.parquet"],
        output_dir=str(tmp_path / "runs"),
        rollout="hf",
    )
    assert config["actor_rollout_ref"]["model"]["trust_remote_code"] is True
    assert config["actor_rollout_ref"]["rollout"]["name"] == "hf"
    assert config["custom_reward_function"]["name"] == "compute_score"
    overrides = flatten_overrides(config)
    assert any(item.startswith("data.train_files=") for item in overrides)
    assert any(item.startswith("actor_rollout_ref.model.path=") for item in overrides)


def test_verl_opd_config_has_teachers(tmp_path):
    config = build_verl_opd_config(
        hf_student_path=tmp_path / "student",
        teachers={"math": {"model_path": tmp_path / "teacher_math", "num_replicas": 1}},
        train_files=["train.parquet"],
        val_files=["val.parquet"],
        output_dir=str(tmp_path / "runs"),
        rollout="hf",
    )
    assert config["distillation"]["enabled"] is True
    assert config["distillation"]["teacher_models"]["math"]["key"] == "math"


def test_torch_backend_is_default(debug_model, stub_tokenizer):
    config = RLConfig()
    engine = build_rollout_engine(config, debug_model, stub_tokenizer)
    assert isinstance(engine, TorchRolloutEngine)
    assert engine.generator is None


def test_vllm_backend_guard():
    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("vLLM installed; guard is not exercised")
    from minideepseek.rl.backends.vllm import VLLMGenerator

    with pytest.raises(RuntimeError, match="vLLM is not installed"):
        VLLMGenerator("missing", None)


def test_http_judge_parses_scores(monkeypatch):
    from minideepseek.rl.judge import HTTPJudge

    judge = HTTPJudge(url="http://localhost:8000/v1", model="judge")
    monkeypatch.setattr(
        judge,
        "_request",
        lambda prompt: '{"factuality": 0.9, "relevance": 1.4, "clarity": -0.2}',
    )
    record = GRPORecord(prompt=[{"role": "user", "content": "q"}])
    scores = judge("answer", record)
    assert scores["factuality"] == 0.9
    assert scores["relevance"] == 1.0
    assert scores["clarity"] == 0.0


def test_http_judge_requires_env(monkeypatch):
    from minideepseek.rl.judge import make_judge

    monkeypatch.delenv("MINIDEEPSEEK_JUDGE_URL", raising=False)
    monkeypatch.delenv("MINIDEEPSEEK_JUDGE_MODEL", raising=False)
    settings = __import__(
        "minideepseek.config", fromlist=["RewardSettings"]
    ).RewardSettings(kind="world", world_judge="http")
    with pytest.raises(RuntimeError, match="world_judge=http requires"):
        make_judge(settings)


def test_lm_eval_guard():
    if importlib.util.find_spec("lm_eval") is not None:
        pytest.skip("lm-eval installed; guard is not exercised")
    from minideepseek.eval.lm_eval import run_lm_eval

    with pytest.raises(RuntimeError, match="lm-evaluation-harness is not installed"):
        run_lm_eval("missing", ["mmlu"])


def test_weight_version_and_generator_consistency(debug_model, stub_tokenizer):
    from minideepseek.rl.rollout import RolloutSample, TorchRolloutEngine

    class StubGenerator:
        model_path = "stub"
        version = 0

        def update_weights(self, state_dict, out_dir):
            self.version += 1
            return self.version

    sample = RolloutSample(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        sampling_logprobs=[0.0, 0.0],
        temperature=1.0,
    )
    engine = TorchRolloutEngine(debug_model, stub_tokenizer, generator=StubGenerator())
    version = engine.sync_weights(debug_model.state_dict())
    assert version == 1
    actor = engine.flat_logprobs([sample])
    sample.sampling_logprobs = actor
    assert engine.check_generator_consistency([sample]) == pytest.approx(0.0, abs=1e-6)
