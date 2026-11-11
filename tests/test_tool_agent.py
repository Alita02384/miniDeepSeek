from __future__ import annotations

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
from minideepseek.model import build_model
from minideepseek.model.wrapper import GenerationOutput
from minideepseek.rl.agent_rollout import ToolRolloutEngine
from minideepseek.rl.grpo import GRPOTrainer
from minideepseek.rl.reward import make_reward_fn, tool_reward
from minideepseek.rl.tools import parse_tool_calls, safe_math_eval


class _Tokenizer:
    vocab_size = 2048
    sha256 = "stub"
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    stop_token_ids = [2, 0]

    def render_chat(self, messages, *, add_generation_prompt=False, tools=None, open_thinking=False):
        del tools, open_thinking
        text = "".join(f"<|{m.get('role')}|>{m.get('content', '')}" for m in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3 + (ord(char) % 2044) for char in text] or [3]

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr((int(value) - 3) % 2044) for value in ids)


class _Generator:
    def __init__(self, tokenizer: _Tokenizer):
        self.tokenizer = tokenizer
        self.calls = 0

    def generate(self, prompts, **kwargs):
        del kwargs
        self.calls += 1
        text = (
            '<tool_call>{"name": "calculate_math", "arguments": {"expression": "1+1"}}</tool_call>'
            if self.calls == 1
            else "2"
        )
        ids = self.tokenizer.encode(text)
        return GenerationOutput(
            sequences=[list(prompts[0]) + ids],
            prompt_lengths=[len(prompts[0])],
            response_ids=[ids],
            response_logprobs=[[-0.1] * len(ids)],
            finished=[True],
            temperature=1.0,
        )


def _tool_record() -> GRPORecord:
    return GRPORecord(
        prompt=[
            {
                "role": "system",
                "content": "",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "calculate_math",
                            "description": "calc",
                            "parameters": {
                                "type": "object",
                                "properties": {"expression": {"type": "string"}},
                                "required": ["expression"],
                            },
                        },
                    }
                ],
            },
            {"role": "user", "content": "1+1"},
        ],
        domain="tool",
        verifier="tool",
        multi_turn=True,
        gt=["2"],
        reference_answer="2",
    )


def test_tool_helpers():
    assert safe_math_eval("1+2*3") == 7
    calls = parse_tool_calls('<tool_call>{"name": "x", "arguments": {}}</tool_call>')
    assert calls == [{"name": "x", "arguments": {}}]


def test_tool_reward_gt_and_alignment():
    record = _tool_record()
    record.rubric["tool_turns"] = [
        '<tool_call>{"name": "calculate_math", "arguments": {"expression": "1+1"}}</tool_call>',
        "2",
    ]
    result = tool_reward("2", record, RewardSettings(kind="tool"))
    assert result.components["tool_alignment"] == 0.5
    assert result.components["gt_match"] == 2.5
    assert result.total == 3.0


def test_tool_rollout_masks_observations():
    tokenizer = _Tokenizer()
    engine = ToolRolloutEngine(None, tokenizer, generator=_Generator(tokenizer), max_turns=3, thinking_ratio=0.0)
    group = engine.rollout([_tool_record()], group_size=1, max_new_tokens=64, temperature=1.0, seed=1)[0]
    sample = group.samples[0]
    assert len(sample.record.rubric["tool_turns"]) == 2
    assert sample.record.rubric["unfinished"] is False
    assert sample.response_mask().count(False) > 0  # tool observation is masked
    assert sample.num_loss_tokens < sample.response_length


def test_tool_grpo_one_step(tmp_path):
    tokenizer = _Tokenizer()
    torch.manual_seed(0)
    model = build_model("debugmodel", seq_len=512, n_mtp_layers=0, device="cpu", swiglu_limit=10.0)
    config = RLConfig(
        model=ModelSettings(
            flavor="debugmodel",
            seq_len=512,
            n_mtp_layers=0,
            device="cpu",
            dtype="float32",
            vocab_limit=2048,
            seed=0,
        ),
        optim=OptimizerSettings(lr=1e-4),
        schedule=ScheduleSettings(warmup_steps=0, decay="constant", min_lr_ratio=1.0),
        train=TrainSettings(steps=1, log_every=1, out_dir=str(tmp_path / "tool_grpo")),
        rl=RLSettings(
            method="grpo",
            group_size=2,
            prompts_per_step=1,
            max_new_tokens=64,
            temperature=1.0,
            clip=0.2,
            max_turns=3,
        ),
        reward=RewardSettings(kind="tool", length_bonus=0.5, think_bonus=1.0, repetition_penalty=0.5, reward_clip=3.0),
    )
    engine = ToolRolloutEngine(None, tokenizer, generator=_Generator(tokenizer), max_turns=3, thinking_ratio=0.0)
    trainer = GRPOTrainer(config, model, tokenizer, [_tool_record()], make_reward_fn(config.reward), engine=engine)
    metrics = trainer.step()
    assert metrics["response_tokens"] > 0
    assert metrics["reward_gt_match"] > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_tool_reward_does_not_use_gt_count_as_expected_calls():
    record = _tool_record()
    record.gt = ["2", "3"]  # two answer values, still one weather/math call
    record.rubric["tool_turns"] = [
        '<tool_call>{"name": "calculate_math", "arguments": {"expression": "1+1"}}</tool_call>',
        "2",
    ]
    result = tool_reward("2", record, RewardSettings(kind="tool"))
    assert result.components["tool_alignment"] == 0.5
    assert result.components["gt_match"] == 1.25  # one of two gt values


def test_tool_reward_unfinished_skips_gt_and_shaping():
    from minideepseek.rl.reward import dispatch_reward

    record = _tool_record()
    record.rubric["tool_turns"] = [
        '<tool_call>{"name": "calculate_math", "arguments": {"expression": "1+1"}}</tool_call>',
        "2",
    ]
    record.rubric["unfinished"] = True
    settings = RewardSettings(
        kind="tool",
        length_bonus=0.5,
        think_bonus=1.0,
        repetition_penalty=0.5,
        reward_clip=3.0,
    )
    result = dispatch_reward("2", record, settings)
    assert "gt_match" not in result.components
    assert "length" not in result.components
    assert "think" not in result.components
    # valid call bonus + unfinished penalty
    assert abs(result.total) < 1e-9


def test_tool_reward_does_not_scan_think_text_for_gt():
    record = _tool_record()
    record.gt = ["42"]
    record.rubric["tool_turns"] = [
        '<tool_call>{"name": "calculate_math", "arguments": {"expression": "1+1"}}</tool_call>',
        "<think>42</think>wrong",
    ]
    result = tool_reward("<think>42</think>wrong", record, RewardSettings(kind="tool"))
    assert result.components["gt_match"] == 0.0


def test_unit_converter_uses_affine_temperature_formulas():
    from minideepseek.rl.tools import execute_tool

    celsius = execute_tool(
        "unit_converter",
        {"value": 10, "from_unit": "celsius", "to_unit": "fahrenheit"},
    )
    fahrenheit = execute_tool(
        "unit_converter",
        {"value": 18, "from_unit": "fahrenheit", "to_unit": "celsius"},
    )
    assert celsius == {"result": 50.0}
    assert fahrenheit == {"result": -7.7778}
