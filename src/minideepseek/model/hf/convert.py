"""State-dict key mapping between training checkpoints and HF directories."""

from __future__ import annotations

import torch

HF_PREFIX = "model."


def training_to_hf_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add the ``model.`` prefix expected by the HF ``PreTrainedModel``."""
    return {
        key if key.startswith(HF_PREFIX) else f"{HF_PREFIX}{key}": value
        for key, value in state.items()
    }


def hf_to_training_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the HF prefix so the tensors can load into ``DeepSeekV4Model``."""
    return {
        key[len(HF_PREFIX) :] if key.startswith(HF_PREFIX) else key: value
        for key, value in state.items()
    }
