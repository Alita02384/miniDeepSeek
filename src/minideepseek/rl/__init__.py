"""Reinforcement learning: rollout, rewards, GRPO, and OPD."""

from .grpo import GRPOTrainer, build_grpo_trainer
from .opd import OPDTrainer, build_opd_trainer, load_teacher
from .reward import (
    RewardResult,
    code_reward,
    dispatch_reward,
    group_advantages,
    make_reward_fn,
    math_reward,
    world_reward,
)
from .rollout import RolloutGroup, RolloutSample, TorchRolloutEngine

__all__ = [
    "GRPOTrainer",
    "OPDTrainer",
    "RewardResult",
    "RolloutGroup",
    "RolloutSample",
    "TorchRolloutEngine",
    "build_grpo_trainer",
    "build_opd_trainer",
    "code_reward",
    "dispatch_reward",
    "group_advantages",
    "load_teacher",
    "make_reward_fn",
    "math_reward",
    "world_reward",
]
