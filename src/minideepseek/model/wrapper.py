"""High-level model wrapper used by every miniDeepSeek component.

The V4 architecture lives in :mod:`minideepseek.model.model` (a self-contained
refactor of the reference implementation).  This wrapper adds the pieces the
rest of the project needs:

* a packed, multi-document batch API (``PackedBatch``);
* a loss that handles response masks and MTP depths;
* per-token log-probabilities for GRPO / OPD;
* a small, torch-only rollout loop (no KV cache yet; correctness first);
* parameter accounting used by the roadmap's calibration checks.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelSettings
from .batch import PackedBatch, shift_targets  # noqa: E402
from .config import UNSET, build_model_config, build_model_config_from_settings


@dataclass
class LossOutput:
    """Weighted training loss and its components."""

    total: Tensor
    main: Tensor
    mtp: dict[int, Tensor] = field(default_factory=dict)
    main_token_count: int = 0
    mtp_token_count: dict[int, int] = field(default_factory=dict)
    indexer: Tensor | None = None

    @property
    def token_count(self) -> int:
        """All prediction slots (main head plus MTP depths), not corpus tokens."""
        return self.main_token_count + sum(self.mtp_token_count.values())


@dataclass
class GenerationOutput:
    """Result of a rollout/generation call."""

    sequences: list[list[int]]
    prompt_lengths: list[int]
    response_ids: list[list[int]]
    response_logprobs: list[list[float]]
    finished: list[bool]
    temperature: float

    def response_text_offsets(self) -> list[int]:
        return self.prompt_lengths


class MiniDeepSeekModel:
    """A thin training/inference facade around ``DeepSeekV4Model``."""

    def __init__(
        self,
        config,
        *,
        device: torch.device | str = "cpu",
        init_weights: bool = True,
        dtype: str | torch.dtype | None = None,
        activation_checkpointing: bool = False,
    ):
        from .model import DeepSeekV4Model

        self.config = config
        self.model: nn.Module = DeepSeekV4Model(config)
        if init_weights:
            self.model.init_weights()
        self.model.to(device)
        self.model.grad_checkpointing = bool(activation_checkpointing)
        self.device = torch.device(device)
        self.autocast_dtype = _resolve_autocast_dtype(dtype, self.device)

    # -- nn.Module delegation -------------------------------------------------
    def __call__(self, batch: PackedBatch, **kwargs):
        return self.forward(batch, **kwargs)

    def train(self) -> MiniDeepSeekModel:
        self.model.train()
        return self

    def eval(self) -> MiniDeepSeekModel:
        self.model.eval()
        return self

    def to(self, device: torch.device | str) -> MiniDeepSeekModel:
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        return self.model.named_parameters()

    def buffers(self) -> Iterator[Tensor]:
        return self.model.buffers()

    def named_buffers(self) -> Iterator[tuple[str, Tensor]]:
        return self.model.named_buffers()

    def modules(self) -> Iterator[nn.Module]:
        return self.model.modules()

    def named_modules(self) -> Iterator[tuple[str, nn.Module]]:
        return self.model.named_modules()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.model.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    @property
    def training(self) -> bool:
        return self.model.training

    # -- properties -----------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return int(self.config.vocab_size)

    @property
    def max_context_length(self) -> int:
        return int(self.config.max_context_length)

    @property
    def n_mtp_layers(self) -> int:
        return int(self.config.n_mtp_layers)

    @property
    def dim(self) -> int:
        return int(self.config.dim)

    # -- forward --------------------------------------------------------------
    def attention_metadata(self, batch: PackedBatch):
        return self.model._get_attention_metadata(  # noqa: SLF001 - upstream hook
            batch.positions,
            padding_mask=batch.padding_mask,
            max_num_documents=batch.num_documents,
            max_context_length=self.max_context_length,
        )

    def _indexer_aux_enabled(self) -> bool:
        for layer in self.config.layers:
            if getattr(layer.attention, "use_indexer_aux", False):
                return True
        return False

    def _aux_loss_denominators(self, batch: PackedBatch) -> Tensor | None:
        if not (self.training and self._indexer_aux_enabled()):
            return None
        counts = [batch.num_tokens]
        for depth in range(1, self.n_mtp_layers + 1):
            counts.append(max(batch.num_tokens - depth, 1))
        return torch.tensor(counts, dtype=torch.float32, device=self.device)

    def forward(self, batch: PackedBatch, *, use_mtp: bool = True) -> tuple[Tensor, ...]:
        """Run the decoder and return one logits tensor per MTP depth."""
        if batch.input_ids.device != self.device:
            batch = batch.to(self.device)
        metadata = self.attention_metadata(batch)
        with _autocast(self.device, self.autocast_dtype):
            outputs = self.model(
                batch.input_ids,
                batch.positions,
                metadata,
                padding_mask=batch.padding_mask,
                aux_loss_denominators=self._aux_loss_denominators(batch),
            )
        outputs = (outputs,) if isinstance(outputs, Tensor) else tuple(outputs)
        if not use_mtp:
            outputs = outputs[:1]
        return outputs

    # -- losses ---------------------------------------------------------------
    @staticmethod
    def _masked_ce(
        logits: Tensor, batch: PackedBatch, shift: int
    ) -> tuple[Tensor, int]:
        targets, valid = shift_targets(batch, shift)
        if targets.numel() == 0 or not bool(valid.any()):
            return logits.new_zeros(()), 0
        selected = logits[: targets.numel()][valid].float()
        loss = F.cross_entropy(selected, targets[valid], reduction="mean")
        return loss, int(valid.sum())

    def compute_loss(
        self,
        batch: PackedBatch,
        *,
        mtp_weight: float = 0.3,
    ) -> LossOutput:
        """Cross-entropy over the main depth plus every configured MTP depth."""
        preds = self.forward(batch)
        main_loss, main_count = self._masked_ce(preds[0], batch, 0)
        total = main_loss
        indexer_loss = self.model.collect_indexer_loss()
        mtp_losses: dict[int, Tensor] = {}
        mtp_counts: dict[int, int] = {}
        depths = len(preds) - 1
        if depths > 0 and mtp_weight > 0:
            per_depth = mtp_weight / depths
            for depth, logits in enumerate(preds[1:], start=1):
                loss, count = self._masked_ce(logits, batch, depth)
                mtp_losses[depth] = loss
                mtp_counts[depth] = count
                total = total + loss * per_depth
        if indexer_loss is not None:
            # The indexer loss is normalised by local tokens.  FSDP/DDP
            # averages parameter gradients across DP ranks, which already
            # turns that into the global per-token value; dividing by the DP
            # size here would under-weight it by another factor of world_size.
            total = total + indexer_loss
        return LossOutput(
            total=total,
            main=main_loss,
            mtp=mtp_losses,
            main_token_count=main_count,
            mtp_token_count=mtp_counts,
            indexer=indexer_loss,
        )

    # -- log-probabilities ----------------------------------------------------
    def token_logprobs(
        self,
        batch: PackedBatch,
        *,
        requires_grad: bool = False,
        temperature: float = 1.0,
    ) -> Tensor:
        """Per-token log-probability of the *observed* next token.

        Element ``t`` is ``log p(input_ids[t] | input_ids[:t])``; element 0 is
        always 0 because there is no predecessor inside the batch.  ``temperature``
        must match the rollout sampling temperature for GRPO ratios.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            logits = self.forward(batch, use_mtp=False)[0]
            log_probs = F.log_softmax(logits[:-1].float() / temperature, dim=-1)
            targets = batch.input_ids[1:]
            values = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            out = torch.zeros(batch.num_tokens, dtype=torch.float32, device=logits.device)
            out[1:] = values
        return out

    def logits_at(
        self,
        batch: PackedBatch,
        positions: Tensor,
        *,
        temperature: float = 1.0,
    ) -> Tensor:
        """Main-depth logits only at ``positions`` (``[N, vocab]``).

        The RL/OPD losses only supervise response tokens.  Computing the LM
        head for every packed token materialises ``[T, 6400]`` (hundreds of MiB
        per micro-batch) and wastes a third of the forward; slice the hidden
        states first.
        """
        core = self._core()
        if positions.numel() == 0:
            return core.tok_embeddings.weight.new_zeros((0, self.vocab_size))
        if batch.input_ids.device != self.device:
            batch = batch.to(self.device)
        # FSDP2 shards ``lm_head.weight`` as a DTensor; slicing the hidden
        # states produces a plain tensor and mixing the two in F.linear is an
        # error.  Pay the full-vocabulary forward on that path only.
        head = getattr(core, "lm_head", None)
        head_weight = getattr(head, "weight", None)
        if head_weight is not None and type(head_weight).__name__ == "DTensor":
            logits = self.full_logits(batch, temperature=temperature)
            return logits[positions.to(logits.device)]
        metadata = self.attention_metadata(batch)
        # RL/OPD never backpropagates the CSA indexer auxiliary loss; computing
        # its teacher in every actor forward costs a chunked [T, K, D] pass for
        # nothing.  Disable it for this forward only.
        saved_aux = []
        for layer in getattr(core, "layers", {}).values():
            attention = getattr(layer, "attention", None)
            aux = getattr(attention, "indexer_aux", None) if attention is not None else None
            if aux is not None:
                saved_aux.append((attention, aux))
                attention.indexer_aux = None
        try:
            with _autocast(self.device, self.autocast_dtype):
                hidden = self.model(
                    batch.input_ids,
                    batch.positions,
                    metadata,
                    padding_mask=batch.padding_mask,
                    aux_loss_denominators=None,
                    use_mtp=False,
                    return_hidden=True,
                )
                selected = hidden[positions.to(hidden.device)]
                logits = head(selected) if head is not None else selected
        finally:
            for attention, aux in saved_aux:
                attention.indexer_aux = aux
        return logits if temperature == 1.0 else logits / temperature

    def response_logprobs(
        self,
        batch: PackedBatch,
        response_mask: Tensor,
        *,
        requires_grad: bool = False,
        temperature: float = 1.0,
    ) -> Tensor:
        """Log-probabilities at response-token positions (``response_mask`` is ``[T]``)."""
        positions = response_mask.nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        if int(positions.min()) == 0:
            raise ValueError("response token at position 0 has no prediction logit")
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            logits = self.logits_at(batch, positions - 1, temperature=temperature)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            targets = batch.input_ids[positions]
            return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def full_logits(self, batch: PackedBatch, *, temperature: float = 1.0) -> Tensor:
        """Main-depth logits (``[T, vocab]``) without MTP."""
        logits = self.forward(batch, use_mtp=False)[0]
        return logits if temperature == 1.0 else logits / temperature

    # -- incremental prefill/decode (rollout) --------------------------------
    def _core(self):
        """Unwrap DDP-style containers to the module that owns ``decode``."""
        module = self.model
        while hasattr(module, "module") and not hasattr(module, "decode"):
            module = module.module
        return module

    @torch.no_grad()
    def prefill(
        self,
        prompt_ids: list[int] | torch.Tensor,
        cache,
        slot: int,
    ) -> torch.Tensor:
        """Run one prompt through the model, populate ``cache[slot]``.

        Returns the main-depth logits ``[T, vocab]``; the caller only needs the
        last position but keeping the whole tensor costs little for prompts.
        """
        from .attention import PackedAttentionMetadata

        self.eval()
        if isinstance(prompt_ids, torch.Tensor):
            tokens = prompt_ids.to(self.device, dtype=torch.long)
        else:
            tokens = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        length = int(tokens.numel())
        if length <= 0:
            raise ValueError("prompt must contain at least one token")
        if length > self.max_context_length:
            raise ValueError(
                f"prompt length {length} exceeds context {self.max_context_length}"
            )
        positions = torch.arange(length, dtype=torch.long, device=self.device)
        metadata = PackedAttentionMetadata(
            cu_seqlens=torch.tensor([0, length], dtype=torch.int32, device=self.device),
            max_seqlen=length,
        )
        with _autocast(self.device, self.autocast_dtype):
            logits = self._core()(
                tokens,
                positions,
                metadata,
                cache=cache,
                slot=slot,
                use_mtp=False,
            )
        return logits

    @torch.no_grad()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """One token per slot; returns ``[B, vocab]`` main-depth logits."""
        with _autocast(self.device, self.autocast_dtype):
            return self._core().decode(input_ids, positions, cache, slots)

    # -- generation -----------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        greedy: bool = False,
        stop_token_ids: Iterable[int] | None = None,
        seed: int | None = None,
    ) -> GenerationOutput:
        """Batched, document-packed rollout (full recompute, no KV cache).

        This is the correctness-first rollout path from roadmap section 6.  It
        computes the whole sequence for every decode step so that the log-probs
        are directly comparable with a training forward pass.
        """
        if not prompts:
            raise ValueError("at least one prompt is required")
        if any(len(p) == 0 for p in prompts):
            raise ValueError("prompts must not be empty")
        longest_prompt = max(len(p) for p in prompts)
        if longest_prompt >= self.max_context_length:
            raise ValueError(
                f"prompt length {longest_prompt} leaves no room in context "
                f"{self.max_context_length}"
            )
        max_new_tokens = min(
            int(max_new_tokens), self.max_context_length - longest_prompt
        )
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must leave room for at least one token")
        device = self.device
        stop = set(int(t) for t in (stop_token_ids or ()))
        was_training = self.training
        self.eval()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        sequences = [list(p) for p in prompts]
        prompt_lengths = [len(p) for p in prompts]
        responses: list[list[int]] = [[] for _ in prompts]
        response_logprobs: list[list[float]] = [[] for _ in prompts]
        finished = [False] * len(prompts)
        eff_temperature = 1.0 if greedy else float(temperature)
        if eff_temperature <= 0:
            raise ValueError("temperature must be positive")

        try:
            for _ in range(max_new_tokens):
                if all(finished):
                    break
                batch = _from_sequences_on_device(sequences, device)
                logits = self.forward(batch, use_mtp=False)[0]
                last = batch.last_token_indices()
                step_logits = logits[last].float()
                next_ids, next_logprobs = _sample_tokens(
                    step_logits,
                    temperature=eff_temperature,
                    top_k=top_k,
                    top_p=top_p,
                    greedy=greedy,
                    generator=generator,
                )
                for index, token in enumerate(next_ids.tolist()):
                    if finished[index]:
                        continue
                    sequences[index].append(int(token))
                    responses[index].append(int(token))
                    response_logprobs[index].append(float(next_logprobs[index].item()))
                    if token in stop:
                        finished[index] = True
        finally:
            if was_training:
                self.train()

        return GenerationOutput(
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            response_ids=responses,
            response_logprobs=response_logprobs,
            finished=finished,
            temperature=eff_temperature,
        )

    # -- sparse-index schedule ------------------------------------------------
    def set_index_topk(self, topk: int) -> int:
        """Set CSA index top-k at runtime; returns the previous value."""
        if topk < 1:
            raise ValueError("index_topk must be positive")
        previous = None
        for layer in self.model.layers.values():
            attention = layer.attention
            if hasattr(attention, "index_topk"):
                if previous is None:
                    previous = int(attention.index_topk)
                attention.index_topk = int(topk)
        return int(previous) if previous is not None else int(topk)

    def current_index_topk(self) -> int | None:
        for layer in self.model.layers.values():
            attention = layer.attention
            if hasattr(attention, "index_topk"):
                return int(attention.index_topk)
        return None

    # -- reporting ------------------------------------------------------------
    def parameter_report(self) -> dict[str, float]:
        """Total and active parameter counts, split by component.

        ``moe_*`` covers the main transformer layers only; ``mtp`` is the whole
        MTP depth stack, including its experts.  This keeps the buckets
        additive (no double counting) while ``active_estimate`` follows the
        project convention of counting MTP routed experts at the same
        ``top_k / num_experts`` sparsity ratio as the main stack.
        """
        total = sum(p.numel() for p in self.model.parameters())

        def _moe_parts(modules) -> tuple[int, int, int]:
            routed = router = shared = 0
            for module in modules:
                routed += sum(p.numel() for p in module.routed_experts.parameters())
                router += sum(p.numel() for p in module.router.parameters())
                if getattr(module, "shared_experts", None) is not None:
                    shared += sum(p.numel() for p in module.shared_experts.parameters())
            return routed, router, shared

        main_routed, main_router, main_shared = _moe_parts(
            layer.feed_forward for layer in self.model.layers.values()
        )
        mtp_routed, _mtp_router, _mtp_shared = _moe_parts(
            layer.feed_forward for layer in self.model.mtp_layers
        )
        # Attention owns most of what remains; keep it as one bucket.
        embedding = self.model.tok_embeddings.weight.numel()
        lm_head = (
            0
            if self.model.lm_head is None or self.config.tie_word_embeddings
            else self.model.lm_head.weight.numel()
        )
        mtp = sum(p.numel() for p in self.model.mtp_layers.parameters())
        other = total - embedding - lm_head - main_routed - main_router - main_shared - mtp
        first_ffn = self.model.layers["0"].feed_forward
        top_k = int(first_ffn.router.top_k)
        n_experts = int(first_ffn.num_experts)
        active = total - (main_routed + mtp_routed) * (1.0 - top_k / max(n_experts, 1))
        return {
            "total": float(total),
            "active_estimate": float(active),
            "embedding": float(embedding),
            "lm_head": float(lm_head),
            "attention_and_other": float(other),
            "moe_routed": float(main_routed),
            "moe_router": float(main_router),
            "moe_shared": float(main_shared),
            "mtp": float(mtp),
        }


def build_model(
    source: str | ModelSettings = "mini_220m_a80m",
    *,
    seq_len: int | None = None,
    max_context: int | None = None,
    n_mtp_layers: int | None = None,
    device: torch.device | str = "cpu",
    tie_word_embeddings: bool | None = None,
    swiglu_limit: Any = UNSET,
    index_topk: int | None = None,
    indexer_loss_coeff: Any = UNSET,
    init_weights: bool = True,
    dtype: str | torch.dtype | None = None,
    activation_checkpointing: bool = False,
) -> MiniDeepSeekModel:
    """Build a miniDeepSeek model from a bundled flavor or a resolved spec.

    Passing a :class:`~minideepseek.config.ModelSettings` is the training path:
    the architecture and run settings were already merged from ``model_file``.
    A flavor string is a convenience for CLIs/tests and loads
    ``configs/model/<flavor>.yaml``.
    """
    if isinstance(source, ModelSettings):
        settings = source
        updates: dict[str, Any] = {}
        if seq_len is not None:
            updates["seq_len"] = int(seq_len)
        if max_context is not None:
            updates["max_context"] = int(max_context)
        elif seq_len is not None:
            updates["max_context"] = int(seq_len)
        if n_mtp_layers is not None:
            updates["n_mtp_layers"] = int(n_mtp_layers)
        if tie_word_embeddings is not None:
            updates["tie_word_embeddings"] = bool(tie_word_embeddings)
        if swiglu_limit is not UNSET:
            updates["swiglu_limit"] = swiglu_limit
        if index_topk is not None:
            updates["index_topk"] = int(index_topk)
        if indexer_loss_coeff is not UNSET:
            updates["indexer_loss_coeff"] = float(indexer_loss_coeff)
        if updates:
            settings = dataclasses.replace(settings, **updates)
        config = build_model_config_from_settings(settings)
        if dtype is None:
            dtype = settings.dtype
    else:
        config = build_model_config(
            source,
            seq_len=seq_len,
            max_context=max_context,
            n_mtp_layers=n_mtp_layers,
            tie_word_embeddings=tie_word_embeddings,
            swiglu_limit=swiglu_limit,
            index_topk=index_topk,
            indexer_loss_coeff=indexer_loss_coeff,
        )
    return MiniDeepSeekModel(
        config,
        device=device,
        init_weights=init_weights,
        dtype=dtype,
        activation_checkpointing=activation_checkpointing,
    )


def model_config_dict(config) -> dict:
    """JSON-serialisable subset of a model config."""
    fields = dataclasses.fields(config)
    out: dict = {}
    for field_info in fields:
        value = getattr(config, field_info.name)
        if field_info.name == "layers" or field_info.name == "mtp_layers":
            continue
        if dataclasses.is_dataclass(value):
            continue
        if isinstance(value, (int, float, str, bool, type(None), tuple, list)):
            out[field_info.name] = value
        else:
            out[field_info.name] = repr(value)
    out["n_layers"] = len(config.layers)
    return out


def _from_sequences_on_device(sequences: list[list[int]], device: torch.device) -> PackedBatch:
    from .batch import from_sequences

    return from_sequences(sequences).to(device)


def _sample_tokens(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    """Sample one token per row and return its log-probability."""
    scaled = logits / temperature
    if greedy:
        log_probs = F.log_softmax(scaled, dim=-1)
        ids = scaled.argmax(dim=-1)
        return ids, log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    filtered = scaled
    if top_k and top_k > 0:
        k = min(int(top_k), filtered.shape[-1])
        threshold = torch.topk(filtered, k, dim=-1).values[..., -1:]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_ids = torch.sort(filtered, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative - probs > float(top_p)
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf")).scatter(
            -1, sorted_ids, sorted_logits
        )
    probs = F.softmax(filtered, dim=-1)
    ids = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    # Sampling may be top-k/top-p filtered, but the policy log-probability used
    # for GRPO ratios must come from the unfiltered model distribution.
    log_probs = F.log_softmax(scaled, dim=-1)
    return ids, log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)


def _resolve_autocast_dtype(dtype, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or dtype in (None, "float32", torch.float32):
        return None
    if isinstance(dtype, str):
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        return mapping.get(dtype.lower())
    return dtype if dtype in (torch.bfloat16, torch.float16) else None


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None or device.type != "cuda":
        return _null_context()
    return torch.autocast(device_type="cuda", dtype=dtype)
