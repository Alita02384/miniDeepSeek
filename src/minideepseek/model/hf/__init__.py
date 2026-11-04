"""Hugging Face format adapter (optional dependency: ``transformers``)."""

from .configuration_minideepseek import MiniDeepSeekConfig
from .modeling_minideepseek import MiniDeepSeekForCausalLM

__all__ = ["MiniDeepSeekConfig", "MiniDeepSeekForCausalLM"]
