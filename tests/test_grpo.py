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


def test_math_extraction_handles_connective_prefixes():
    assert extract_answer("因此答案是 42") == "42"
    assert extract_answer("所以答案是 42") == "42"
    assert extract_answer("因此 x = 42") == "42"
    assert extract_answer("答案是 42 元") == "42"
    assert math_correct("因此答案是 42", "42")[0]


def test_grpo_resume_does_not_override_rl_lr(tmp_path, stub_tokenizer):
    from minideepseek.model import build_model
    from minideepseek.model.checkpoint import save_checkpoint
    from minideepseek.train.optimizer import build_optimizer

    source = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu")
    optimizer = build_optimizer(source, OptimizerSettings(lr=2e-5))
    checkpoint = tmp_path / "sft"
    save_checkpoint(checkpoint, source, optimizer=optimizer, step=1)
    records = [GRPORecord(prompt=[{"role": "user", "content": "q"}], domain="math")]
    config = _rl_config(tmp_path)
    config.optim.lr = 2e-6
    config.train.resume = str(checkpoint)
    trainer, _model = build_grpo_trainer(
        config,
        records,
        lambda text, _record: RewardResult(total=0.0),
        tokenizer=stub_tokenizer,
    )
    assert all(
        group["base_lr"] == 2e-6 for group in trainer.optimizer.param_groups
    )


def test_grpo_ddp_all_reduces_gradients(tmp_path, stub_tokenizer, monkeypatch):
    import minideepseek.rl.grpo as grpo_module
    from minideepseek.model import build_model
    from minideepseek.train.distributed import DistributedContext

    calls = []
    monkeypatch.setattr(
        grpo_module,
        "all_reduce_gradients",
        lambda model, world_size: calls.append(world_size),
    )
    records = [GRPORecord(prompt=[{"role": "user", "content": "q"}], domain="math")]
    config = _rl_config(tmp_path)
    model = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu")
    trainer = GRPOTrainer(
        config,
        model,
        stub_tokenizer,
        records,
        lambda text, _record: RewardResult(total=0.0),
        distributed=DistributedContext(enabled=True, mode="ddp", world_size=2),
    )
    trainer.step()
    assert calls == [2]


def test_grpo_prompt_budget_is_global(tmp_path, stub_tokenizer):
    from minideepseek.model import build_model
    from minideepseek.rl.rollout import TorchRolloutEngine
    from minideepseek.train.distributed import DistributedContext

    records = [
        GRPORecord(prompt=[{"role": "user", "content": str(index)}], domain="math")
        for index in range(4)
    ]
    config = _rl_config(tmp_path)
    model = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu")
    trainer = GRPOTrainer(
        config,
        model,
        stub_tokenizer,
        records,
        lambda text, _record: RewardResult(total=0.0),
        engine=TorchRolloutEngine(model, stub_tokenizer),
        distributed=DistributedContext(world_size=2),
    )
    assert len(trainer._sample_records()) == 1


def test_grpo_syncs_generator_after_update(tmp_path, stub_tokenizer):
    from minideepseek.model import build_model
    from minideepseek.rl.rollout import TorchRolloutEngine

    class FakeGenerator:
        def __init__(self, model):
            self.model = model
            self.version = 0

        def generate(self, prompts, **kwargs):
            return self.model.generate(prompts, **kwargs)

        def update_weights(self, state_dict, out_dir):
            del state_dict, out_dir
            self.version += 1

    records = [GRPORecord(prompt=[{"role": "user", "content": "q"}], domain="math")]
    config = _rl_config(tmp_path)
    model = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu")
    generator = FakeGenerator(model)
    engine = TorchRolloutEngine(model, stub_tokenizer, generator=generator)
    trainer = GRPOTrainer(
        config,
        model,
        stub_tokenizer,
        records,
        lambda text, _record: RewardResult(total=0.0),
        engine=engine,
    )
    stats = trainer.step()
    assert generator.version == 1
    assert stats["rollout_weight_version"] == 1
    from minideepseek.train.losses import iter_moe_layers

    assert all(
        float(moe.router.tokens_per_expert_E.sum()) == 0.0
        for _name, moe in iter_moe_layers(trainer.model.model)
    )


def test_rollout_prompt_opens_thinking_and_format_bonus_is_reachable():
    from minideepseek.rl.rollout import TorchRolloutEngine
    from minideepseek.tokenizer import load_tokenizer

    tokenizer = load_tokenizer()
    record = GRPORecord(
        prompt=[{"role": "user", "content": "1+1"}],
        domain="math",
        reference_answer="2",
        verifier="math_exact_match",
    )
    engine = TorchRolloutEngine(None, tokenizer)
    prompt_text = tokenizer.decode(engine.encode_prompt(record))
    assert "<think>" in prompt_text
    assert "</think>" not in prompt_text

    result = math_reward(
        "42</think>", record, RewardSettings(kind="math", format_bonus=0.1)
    )
    assert result.components["format"] == 0.1


def test_general_reward_model_and_shaping():
    from minideepseek.rl.reward import dispatch_reward

    class FakeRewardModel:
        def get_score(self, messages, response):
            del messages, response
            return 1.5

    record = GRPORecord(prompt=[{"role": "user", "content": "hello"}], domain="general")
    settings = RewardSettings(
        kind="general",
        length_bonus=0.5,
        think_bonus=1.0,
        repetition_penalty=0.5,
        reward_clip=3.0,
    )
    result = dispatch_reward(
        "你好，这是回答。<think>思考内容足够长足够长足够长</think>",
        record,
        settings,
        reward_model=FakeRewardModel(),
    )
    assert result.components["reward_model"] == 1.5
    assert result.components["length"] == 0.5
    assert result.components["think"] > 0
    assert result.total <= 3.0
