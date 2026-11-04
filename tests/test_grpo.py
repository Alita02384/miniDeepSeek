from __future__ import annotations

from pathlib import Path

import torch

from minideepseek.config import (
    ModelSettings,
    OptimizerSettings,
    RewardSettings,
    RLConfig,
    RLSettings,
    ScheduleSettings,
    TrainSettings,
)
from minideepseek.data.grpo import GRPORecord
from minideepseek.rl.grpo import GRPOTrainer, build_grpo_trainer
from minideepseek.rl.reward import (
    RewardResult,
    code_reward,
    extract_answer,
    group_advantages,
    math_correct,
    math_reward,
    medical_reward,
    normalize_answer,
    world_reward,
)


def _rl_config(tmp_path, **rl_overrides) -> RLConfig:
    rl = dict(
        method="grpo",
        group_size=2,
        prompts_per_step=2,
        max_new_tokens=3,
        temperature=1.0,
        clip=0.2,
    )
    rl.update(rl_overrides)
    return RLConfig(
        model=ModelSettings(
            flavor="debugmodel",
            seq_len=128,
            n_mtp_layers=0,
            swiglu_limit=10.0,
            device="cpu",
            dtype="float32",
            seed=0,
        ),
        optim=OptimizerSettings(lr=1e-4),
        schedule=ScheduleSettings(warmup_steps=0, decay="constant", min_lr_ratio=1.0),
        train=TrainSettings(steps=1, log_every=1, out_dir=str(tmp_path / "rl")),
        rl=RLSettings(**rl),
        reward=RewardSettings(kind="math"),
    )


def test_math_reward_and_parsing():
    record = GRPORecord(
        prompt=[{"role": "user", "content": "1+1"}],
        reference_answer="2",
        domain="math",
        verifier="math_exact_match",
    )
    result = math_reward("答案是 2", record, RewardSettings(kind="math"))
    assert result.components["correctness"] == 1.0
    assert result.components["parse"] == 0.1
    wrong = math_reward("答案是 3", record, RewardSettings(kind="math"))
    assert wrong.components["correctness"] == 0.0
    assert extract_answer("so the answer is \\boxed{42}") == "42"
    assert normalize_answer(" 1,000. ") == "1000"
    assert math_correct("\\boxed{1/2}", "0.5")[0]


def test_code_reward_pass_and_fail():
    record = GRPORecord(
        prompt=[{"role": "user", "content": "add"}],
        domain="code",
        verifier="unit_test",
        rubric={"tests": "assert add(1, 2) == 3"},
    )
    settings = RewardSettings(kind="code", code_timeout_s=5.0)
    passing = code_reward("```python\ndef add(a, b):\n    return a + b\n```", record, settings)
    assert passing.total == 1.0
    failing = code_reward("```python\ndef add(a, b):\n    return a - b\n```", record, settings)
    assert failing.total == 0.0
    broken = code_reward("```python\ndef add(a, b)\n```", record, settings)
    assert broken.total == -0.1


def test_world_and_medical_rewards():
    record = GRPORecord(
        prompt=[{"role": "user", "content": "What is the capital of France?"}],
        reference_answer="Paris is the capital of France",
        domain="world",
        verifier="llm_judge",
    )

    def judge(response, _record):
        return {
            "factuality": 1.0,
            "relevance": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "fluency": 1.0,
            "citation_support": 0.0,
        }

    settings = RewardSettings(kind="world")
    result = world_reward("Paris is the capital of France.", record, settings, judge=judge)
    assert abs(result.total - 0.95) < 1e-6
    assert result.details["rubric"]["factuality"] == 1.0
    medical = GRPORecord(prompt=[{"role": "user", "content": "I feel sick"}], domain="medical")
    safe = medical_reward("请尽快就医并咨询专业医生。", medical, RewardSettings(kind="medical"))
    unsafe = medical_reward("无需就医，多喝水。", medical, RewardSettings(kind="medical"))
    assert safe.components["safety"] == 1.0
    assert unsafe.components["safety"] < 1.0


def test_group_advantages():
    advantages = group_advantages([1.0, 1.0, 1.0])
    assert advantages == [0.0, 0.0, 0.0]
    varied = group_advantages([0.0, 1.0])
    assert varied[0] < 0 < varied[1]
    assert abs(sum(varied)) < 1e-9


def test_grpo_one_step_and_ratio_consistency(tmp_path, stub_tokenizer):
    records = [
        GRPORecord(prompt=[{"role": "user", "content": "q1"}], domain="math", verifier="math_exact_match"),
        GRPORecord(prompt=[{"role": "user", "content": "q2"}], domain="math", verifier="math_exact_match"),
    ]
    config = _rl_config(tmp_path)
    from minideepseek.model import build_model

    model = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu", swiglu_limit=10.0)

    def reward_fn(text, _record):
        value = 1.0 if len(text) % 2 == 0 else 0.0
        return RewardResult(total=value, components={"dummy": value})

    trainer = GRPOTrainer(config, model, stub_tokenizer, records, reward_fn)
    metrics = trainer.train(1)
    assert metrics["steps_run"] == 1
    assert metrics["reward_mean_first"] >= 0.0
    assert abs(metrics["approx_kl_first"]) < 1e-3
    assert metrics["response_tokens_first"] > 0
    assert (Path(config.train.out_dir) / "final" / "meta.json").exists()


def test_build_grpo_trainer_returns_model(tmp_path, stub_tokenizer):
    records = [GRPORecord(prompt=[{"role": "user", "content": "q"}], domain="math")]
    config = _rl_config(tmp_path)
    trainer, model = build_grpo_trainer(
        config, records, lambda text, _record: RewardResult(total=0.0), tokenizer=stub_tokenizer
    )
    assert trainer.model is model
    assert model.vocab_size == 2048


def test_grpo_resume_loads_sft_checkpoint(tmp_path, stub_tokenizer):
    from minideepseek.model import build_model, from_sequences
    from minideepseek.model.checkpoint import save_checkpoint

    torch.manual_seed(7)
    source = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu", swiglu_limit=10.0)
    checkpoint = tmp_path / "sft"
    save_checkpoint(checkpoint, source, tokenizer_hash=stub_tokenizer.sha256)
    records = [GRPORecord(prompt=[{"role": "user", "content": "q"}], domain="math")]
    config = _rl_config(tmp_path)
    config.train.resume = str(checkpoint)
    trainer, model = build_grpo_trainer(
        config, records, lambda text, _record: RewardResult(total=0.0), tokenizer=stub_tokenizer
    )
    batch = from_sequences([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.equal(source.forward(batch)[0], model.forward(batch)[0])


def test_grpo_domain_routing_metrics(tmp_path, stub_tokenizer):
    records = [
        GRPORecord(prompt=[{"role": "user", "content": "q"}], domain="math"),
    ]
    config = _rl_config(tmp_path, log_domain_routing=True)
    from minideepseek.model import build_model

    model = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu", swiglu_limit=10.0)
    trainer = GRPOTrainer(
        config,
        model,
        stub_tokenizer,
        records,
        lambda text, _record: RewardResult(total=0.0),
    )
    metrics = trainer.train(1)
    assert any(key.startswith("routing_domain_math") for key in metrics)
    assert metrics["routing_domain_math_tokens_first"] > 0
