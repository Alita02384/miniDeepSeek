"""Hugging Face ``PreTrainedModel`` adapter for miniDeepSeek.

This is the hand-off format used by verl / vLLM: the module keeps the internal
parameter layout (identical to the training checkpoint) but exposes the HF
``forward`` / ``generate`` contract.  KV cache is intentionally disabled; the
model recomputes the full packed sequence, which matches the torch rollout path
and keeps log-probabilities comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from minideepseek.model.hf.configuration_minideepseek import MiniDeepSeekConfig
from minideepseek.model.model import DeepSeekV4Model


@dataclass
class _PackedInputs:
    tokens: Tensor
    positions: Tensor
    cu_seqlens: Tensor
    keep: Tensor  # [B, T] bool, True for real tokens
    logits_shape: tuple[int, int]
    row_index: Tensor  # [T] row id per packed token
    token_index: Tensor  # [T] column id per packed token


class MiniDeepSeekForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniDeepSeekConfig
    base_model_prefix = "model"
    _no_split_modules = ["DeepSeekV4TransformerBlock", "MTPBlock"]
    _supports_cache_class = False
    _supports_static_cache = False
    _supports_sdpa = False

    def __init__(self, config: MiniDeepSeekConfig):
        super().__init__(config)
        self.model = DeepSeekV4Model(config.to_model_config())
        self.model.init_weights()
        self.post_init()

    # -- HF plumbing ----------------------------------------------------------
    def get_input_embeddings(self) -> nn.Module:
        return self.model.tok_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.tok_embeddings = value

    def get_output_embeddings(self) -> nn.Module | None:
        return self.model.lm_head

    def set_output_embeddings(self, value: nn.Module | None) -> None:
        self.model.lm_head = value

    def tie_weights(self) -> None:
        """The internal model owns tying; HF's generic tying does not apply."""

    def _init_weights(self, module: nn.Module) -> None:
        del module

    # -- packed batching ------------------------------------------------------
    def _pack(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
    ) -> _PackedInputs:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        batch, length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(torch.bool)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.clamp_min(0)
        else:
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)
            if position_ids.shape != input_ids.shape:
                raise ValueError("position_ids must match input_ids")

        tokens: list[Tensor] = []
        positions: list[Tensor] = []
        lengths: list[int] = []
        rows: list[Tensor] = []
        columns: list[Tensor] = []
        for row in range(batch):
            keep = attention_mask[row]
            count = int(keep.sum())
            if count == 0:
                raise ValueError("every batch row needs at least one non-padding token")
            ids = input_ids[row][keep].long()
            pos = position_ids[row][keep].long()
            if int(pos[0]) != 0:
                pos = pos - pos[0]
            tokens.append(ids)
            positions.append(pos)
            lengths.append(count)
            rows.append(torch.full((count,), row, dtype=torch.long, device=input_ids.device))
            columns.append(keep.nonzero(as_tuple=False).flatten())
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device=input_ids.device,
        )
        return _PackedInputs(
            tokens=torch.cat(tokens),
            positions=torch.cat(positions),
            cu_seqlens=cu_seqlens,
            keep=attention_mask,
            logits_shape=(batch, length),
            row_index=torch.cat(rows),
            token_index=torch.cat(columns),
        )

    def _scatter_logits(self, packed_logits: Tensor, packed: _PackedInputs) -> Tensor:
        batch, length = packed.logits_shape
        out = packed_logits.new_zeros(batch, length, packed_logits.shape[-1])
        out[packed.row_index, packed.token_index] = packed_logits
        return out

    # -- forward --------------------------------------------------------------
    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values=None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del use_cache, output_attentions, output_hidden_states, kwargs, inputs_embeds
        if past_key_values is not None:
            raise NotImplementedError(
                "miniDeepSeek uses a full-recompute packed forward; KV cache is disabled"
            )
        if input_ids is None:
            raise ValueError("input_ids is required")
        packed = self._pack(input_ids, attention_mask, position_ids)
        metadata = self.model._get_attention_metadata(packed.positions)
        outputs = self.model(packed.tokens, packed.positions, metadata)
        packed_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        logits = self._scatter_logits(packed_logits.float(), packed)

        loss = None
        if labels is not None:
            loss = self._loss(packed_logits, labels, packed)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def _loss(self, packed_logits: Tensor, labels: Tensor, packed: _PackedInputs) -> Tensor:
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        targets = labels[packed.row_index, packed.token_index]
        # Predict the next token inside each row only.
        next_targets = labels.new_full(targets.shape, -100)
        valid = torch.zeros_like(targets, dtype=torch.bool)
        offset = 0
        for count in torch.diff(packed.cu_seqlens).tolist():
            if count > 1:
                next_targets[offset : offset + count - 1] = targets[offset + 1 : offset + count]
                valid[offset : offset + count - 1] = True
            offset += count
        valid = valid & (next_targets != -100)
        if not bool(valid.any()):
            return packed_logits.new_zeros(())
        return F.cross_entropy(packed_logits[valid], next_targets[valid])

    # -- generation -----------------------------------------------------------
    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        past_key_values=None,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> dict:
        del past_key_values
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.clamp_min(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        del beam_idx
        return past_key_values
