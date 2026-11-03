from __future__ import annotations

import torch

from minideepseek.config import (
    ModelSettings,
    OptimizerSettings,
    RLConfig,
    RLSettings,
    ScheduleSettings,
    TeacherSettings,
    TrainSettings,
)
from minideepseek.data.opd import OPDPrompt
from minideepseek.model import build_model, from_sequences
from minideepseek.model.checkpoint import save_checkpoint
from minideepseek.rl.opd import OPDTrainer, load_teacher


def _opd_config(tmp_path, **rl_overrides) -> RLConfig:
    rl = dict(method="opd", prompts_per_step=2, max_new_tokens=3, temperature=1.0)
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
        train=TrainSettings(steps=1, log_every=1, out_dir=str(tmp_path / "opd")),
        rl=RLSettings(**rl),
    )


def test_opd_single_step_two_teachers(tmp_path, stub_tokenizer):
    torch.manual_seed(1)
    student = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu", swiglu_limit=10.0)
    torch.manual_seed(2)
    math_teacher = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu", swiglu_limit=10.0)
    torch.manual_seed(3)
    code_teacher = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu", swiglu_limit=10.0)
    teachers = {"math": math_teacher, "code": code_teacher}
    prompts = [
        OPDPrompt(prompt=[{"role": "user", "content": "m"}], teacher_key="math", domain="math"),
        OPDPrompt(prompt=[{"role": "user", "content": "c"}], teacher_key="code", domain="code"),
    ]
    teacher_before = [parameter.detach().clone() for parameter in math_teacher.parameters()]
    student_before = [parameter.detach().clone() for parameter in student.parameters()]
    config = _opd_config(tmp_path)
    trainer = OPDTrainer(config, student, stub_tokenizer, prompts, teachers, teacher_weights={"math": 1.0, "code": 1.0})
    metrics = trainer.train(1)
    assert metrics["steps_run"] == 1
    assert metrics["kl_math_first"] >= 0.0
    assert metrics["kl_code_first"] >= 0.0
    assert metrics["response_tokens_first"] > 0
    assert all(torch.equal(before, after) for before, after in zip(teacher_before, math_teacher.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(student_before, student.parameters()))


def test_opd_missing_teacher_raises(tmp_path, stub_tokenizer):
    student = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu", swiglu_limit=10.0)
    prompts = [OPDPrompt(prompt=[{"role": "user", "content": "x"}], teacher_key="world", domain="world")]
    config = _opd_config(tmp_path)
    try:
        trainer = OPDTrainer(config, student, stub_tokenizer, prompts, {"math": student}, teacher_weights={"math": 1.0})
        trainer.step()
    except KeyError as error:
        assert "world" in str(error)
    else:
        raise AssertionError("expected KeyError for a missing teacher")


def test_load_teacher_from_checkpoint(tmp_path):
    torch.manual_seed(4)
    model = build_model("debugmodel", seq_len=128, n_mtp_layers=0, device="cpu", swiglu_limit=10.0)
    checkpoint = tmp_path / "teacher"
    save_checkpoint(checkpoint, model, tokenizer_hash="stub")
    settings = TeacherSettings(name="t", key="math", checkpoint=str(checkpoint), flavor="debugmodel", seq_len=128)
    loaded = load_teacher(settings, tokenizer=type("T", (), {"sha256": "stub"})(), device="cpu")
    batch = from_sequences([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.equal(model.forward(batch)[0], loaded.forward(batch)[0])
