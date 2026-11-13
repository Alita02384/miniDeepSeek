"""Typed YAML configuration objects.

Every stage has a small dataclass tree; YAML files under ``configs/`` map onto
these fields.  Unknown keys are rejected so a typo cannot silently change a
training run, and every run records the resolved config next to its checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import torch
import yaml

from .paths import CONFIG_DIR, PROJECT_ROOT

T = TypeVar("T")


# -- base ---------------------------------------------------------------------
@dataclass
class BaseConfig:
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        return _from_mapping(cls, data)


def _from_mapping(cls: type[T], data: dict[str, Any]) -> T:
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name in field_names:
        if name not in data:
            continue
        kwargs[name] = _convert(hints[name], data[name])
    return cls(**kwargs)


def _convert(annotation: Any, value: Any) -> Any:
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _convert(candidate, value)
            except (TypeError, ValueError):
                continue
        return value
    if origin in (list, tuple, set):
        item_type = args[0] if args else Any
        converted = [_convert(item_type, item) for item in value]
        return origin(converted)
    if origin is dict:
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return {
            _convert(key_type, k): _convert(value_type, v) for k, v in value.items()
        }
    if dataclasses.is_dataclass(annotation):
        return _from_mapping(annotation, value)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"expected bool, got {value!r}")
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        return str(value)
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"top-level YAML must be a mapping: {path}")
    return data


def _model_file_path(value: str | Path, base_dir: Path) -> Path:
    """Resolve a model config path from a training config.

    Paths are accepted relative to the training config, the project root, or
    the current directory (in that order).  A bare role name such as
    ``mini_220m_a80m`` is resolved against ``configs/model/``.
    """
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    for root in (base_dir, PROJECT_ROOT, Path.cwd()):
        path = root / candidate
        if path.exists():
            return path.resolve()
    if candidate.suffix == "":
        bundled = CONFIG_DIR / "model" / f"{candidate.name}.yaml"
        if bundled.exists():
            return bundled.resolve()
    # Return the most useful candidate so ``load_yaml`` reports a concrete path.
    return (base_dir / candidate).resolve()


def _model_field(cls: type) -> bool:
    return "model" in {field_info.name for field_info in dataclasses.fields(cls)}


def resolve_model_file(
    data: dict[str, Any],
    *,
    base_dir: Path,
    override: str | Path | None = None,
) -> dict[str, Any]:
    """Merge a ``model_file`` into the inline ``model`` mapping.

    A training config can either point at an architecture file explicitly::

        model_file: configs/model/mini_220m_a80m.yaml
        model:
          seq_len: 4096

    or select one of the bundled files through ``model.flavor``.  Inline
    ``model`` values always win, which keeps run-specific choices (context
    length, MTP depth, precision, ...) out of the architecture file.
    """
    data = dict(data)
    inline = data.get("model") or {}
    if not isinstance(inline, dict):
        raise TypeError("model config must be a mapping")
    model_file = override or data.get("model_file")
    if model_file is None:
        flavor = inline.get("flavor")
        if flavor:
            bundled = CONFIG_DIR / "model" / f"{flavor}.yaml"
            if bundled.exists():
                model_file = bundled
            elif "dim" not in inline and "n_layers" not in inline:
                raise ValueError(
                    f"model flavor {flavor!r} has no configs/model/{flavor}.yaml; "
                    "set model_file or inline the architecture fields"
                )
    if model_file is None:
        return data
    path = _model_file_path(model_file, base_dir)
    model_data = load_yaml(path)
    if not isinstance(model_data, dict):
        raise TypeError(f"model config must be a mapping: {path}")
    merged = deep_update(model_data, inline)
    # The selected file owns the model identity; allowing an inline ``flavor``
    # to rename another architecture is a silent-mismatch trap.
    if "flavor" in model_data:
        merged["flavor"] = model_data["flavor"]
    data["model"] = merged
    data["model_file"] = str(path)
    return data


def load_config(
    cls: type[T],
    path: str | Path,
    *,
    model_file: str | Path | None = None,
) -> T:
    """Load a typed YAML config, resolving its model architecture file.

    ``model_file`` is a CLI-level override: it replaces the model file named by
    the YAML while keeping the inline ``model`` run settings on top.
    """
    path = Path(path)
    data = load_yaml(path)
    if _model_field(cls):
        data = resolve_model_file(
            data,
            base_dir=path.resolve().parent,
            override=model_file,
        )
    return _from_mapping(cls, data)


def dump_config(config: BaseConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


# -- model --------------------------------------------------------------------
@dataclass
class ModelSettings(BaseConfig):
    """A complete model specification.

    ``configs/model/*.yaml`` files are exactly this dataclass; a training
    config points at one through ``model_file`` (or a bundled ``flavor`` name)
    and may override any field inline.  The defaults reproduce the V4 debug
    model so ad-hoc ``ModelSettings()`` construction stays cheap and testable.
    """

    # Identity and architecture.
    flavor: str = "debugmodel"
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 16
    head_dim: int = 256
    rope_head_dim: int = 32
    q_lora_rank: int = 128
    o_lora_rank: int = 128
    n_groups: int = 2
    compress_ratios: tuple[int, ...] = (0, 0, 4, 128)
    window_size: int = 16
    index_n_heads: int = 8
    index_head_dim: int = 64
    moe_inter_dim: int = 256
    num_experts: int = 4
    num_shared_experts: int = 1
    top_k: int = 3
    n_hash_layers: int = 2
    route_norm: bool = False
    route_scale: float = 1.5
    load_balance_coeff: float | None = 1e-3
    swiglu_limit: float | None = 10.0
    hc_mult: int = 4
    sinkhorn_iters: int = 20
    vocab_size: int = 2048
    tie_word_embeddings: bool = False
    n_mtp_layers: int = 0
    # Training sequence length; ``max_context`` is the RoPE/generation limit.
    seq_len: int = 16384
    max_context: int = 16384
    # None derives ``min(2048, max(seq_len // 4, 64))`` at build time.
    index_topk: int | None = 16
    #: Upper bound on HCA's per-query candidate count.  HCA selects *every*
    #: compressed entry up to the current position, so at 8K (ratio 32) the
    #: kernel gets TOPK=256 -- and attn_gym statically unrolls that loop, which
    #: costs ~1 h to compile.  Capping keeps the most recent entries and reuses
    #: the 4K specialization.  ``None`` keeps the uncapped behaviour.
    hca_topk_max: int | None = None
    indexer_loss_coeff: float = 0.01
    # Runtime settings.
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 1234
    # Debug-only: remap token ids modulo this limit (2048 for the V4 debug model).
    vocab_limit: int | None = None


@dataclass
class OptimizerSettings(BaseConfig):
    kind: str = "hybrid"  # hybrid | adamw | muon
    lr: float = 6e-4
    muon_lr: float | None = None
    adamw_lr: float | None = None
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    # V4 report: AdamW epsilon 1e-20.
    eps: float = 1e-20
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 10
    ns_late_steps: int = 2
    # V4 Algorithm 1 rescales UV^T by sqrt(max(n, m)) * gamma with gamma=0.18.
    muon_update_scale: float = 0.18
    grad_clip: float = 1.0


@dataclass
class ScheduleSettings(BaseConfig):
    warmup_steps: int = 500
    decay: str = "cosine"  # cosine | linear | constant
    min_lr_ratio: float = 0.1
    # V4 keeps the peak LR for most of training and decays near the end.
    # 0.0 reproduces the older "cosine right after warmup" schedule.
    decay_start_ratio: float = 0.0


@dataclass
class DataSourceSettings(BaseConfig):
    path: str
    weight: float = 1.0
    domain: str = "general"


@dataclass
class DataSettings(BaseConfig):
    path: str | None = None
    sources: list[DataSourceSettings] = field(default_factory=list)
    tokenized_path: str | None = None
    tokenizer_dir: str | None = None
    seq_len: int | None = None
    max_records: int | None = None
    max_tokens: int | None = None
    batch_tokens: int = 4096
    # Linear batch-token warmup: start at ``batch_tokens`` and reach
    # ``batch_tokens_final`` after ``batch_warmup_ratio`` of optimizer steps.
    batch_tokens_final: int | None = None
    batch_warmup_ratio: float = 0.0
    shuffle: bool = True
    seed: int = 1234
    domain: str = "general"
    #: With ``data.sources``, keep every optimizer step domain-homogeneous:
    #: one source per step, ``grad_accum`` consecutive micro-batches from it.
    #: ``False`` interleaves sources at micro-batch granularity instead.
    lockstep: bool = True
    #: Re-cut the pack grid to ``seq_len`` tokens at load time.  The streaming
    #: packer is next-fit, so a 2K pack is only ~76% full; this restores the
    #: per-step token budget without touching the tokenised files.  Applied to
    #: the training sources only, so validation stays document-aligned.
    full_packs: bool = False


@dataclass
class AnticipatoryRoutingSettings(BaseConfig):
    enabled: bool = False
    delta: int = 1
    max_steps: int = 100


@dataclass
class DistributedSettings(BaseConfig):
    """Multi-GPU launch policy.

    ``enabled`` is normally inferred from ``torchrun``'s ``WORLD_SIZE``; set it
    explicitly only for tests.  ``fsdp2`` shards transformer blocks + root,
    ``ddp`` replicates the model (useful for small debug runs).
    """

    enabled: bool = False
    mode: str = "fsdp2"  # fsdp2 | ddp
    backend: str = "auto"  # auto | nccl | gloo
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    reshard_after_forward: bool = True
    broadcast_buffers: bool = True


@dataclass
class TrainSettings(BaseConfig):
    #: Curriculum stage index stamped on every metrics row.  ``None`` for a
    #: standalone SFT/RL run.  ``_stamp`` reads ``config.train.stage_index``;
    #: keeping the field here (not only on ``TrainConfig``) is what makes a
    #: standalone run work instead of raising AttributeError on step 1.
    stage_index: int | None = None
    steps: int = 10
    grad_accum: int = 1
    log_every: int = 1
    ckpt_every: int = 0
    #: Number of most-recent ``step_*`` checkpoints to keep; 0 keeps all.
    keep_last_ckpt: int = 0
    #: Save ``best/`` whenever validation loss improves.
    save_best: bool = True
    #: Rewrite ``report.html`` every N steps (0 disables the live report).
    report_every: int = 0
    #: Dump the raw per-router expert counts (and the learned routers' balance
    #: bias) into ``metrics.jsonl`` every N steps (0 disables).  This is the only
    #: routing signal that cannot be rebuilt from the group-level scalars later.
    router_every: int = 0
    #: Up-weight the assistant terminator in the SFT loss.  ``<|im_end|>`` is
    #: ~0.3% of target tokens, so models learn replies but not when to stop.
    #: 1.0 keeps the historical loss exactly; 8-16 makes stopping learnable.
    eos_weight: float = 1.0
    #: Token id of the terminator (the SFT configs set 2 for <|im_end|>).
    eos_token_id: int | None = None
    out_dir: str = "checkpoints/debug"
    mtp_weight: float = 0.3
    mtp_weight_final: float | None = None
    mtp_decay_ratio: float = 0.8
    balance_loss_coeff: float = 1e-4
    #: Per-block activation checkpointing: ~30% slower, much lower peak memory.
    #: Needed for 8K stages (and 220m@4K) on 24 GB cards.
    activation_checkpointing: bool = False
    #: When ``resume`` is unset, continue from the newest ``final``/``step_*``
    #: checkpoint in ``out_dir`` instead of starting from scratch.  This is what
    #: makes "re-run the same command" recover an interrupted run.
    auto_resume: bool = False
    #: Allow a run whose out_dir already has final/ to overwrite it.  Without
    #: this, an explicit ``resume`` that points outside out_dir is rejected so
    #: a data/config change cannot silently continue the old checkpoint.
    force_restart: bool = False
    resume: str | None = None
    # Skip already-consumed micro-batches when resuming the same dataset.  A
    # curriculum stage uses a different dataset, so it sets this to false.
    resume_data: bool = True
    # Load the optimizer state when resuming the same run.  Stage changes
    # (pretrain -> SFT) should start a fresh optimizer and set this false.
    resume_optimizer: bool = True
    # Stage changes (pretrain -> SFT -> RL) start their own step counter.
    resume_step: bool = True
    val_path: str | None = None
    #: Per-domain validation packs.  Each entry is evaluated separately so
    #: ``metrics.jsonl`` carries ``val_loss_<domain>`` next to the aggregate.
    val_sources: list[DataSourceSettings] = field(default_factory=list)
    val_every: int = 0
    val_steps: int = 10
    # Loss-spike stability controls (roadmap section 3.6). Defaults keep the
    # mechanism inert; turn them on for long runs.
    loss_spike_window: int = 100
    loss_spike_sigma: float = 3.0
    loss_spike_min_history: int = 20
    loss_spike_auto_ar: bool = False
    loss_spike_ar_steps: int = 50
    loss_spike_rollback: bool = False
    rollback_every: int = 0
    max_rollbacks: int = 3
    # Sparse-index warmup (roadmap section 3.5); None disables scheduling.
    index_topk_final: int | None = None
    index_topk_warmup_ratio: float = 0.0
    # Explicit shrink window in global (cumulative) steps.  When set it takes
    # precedence over ``index_topk_warmup_ratio``; curriculum stages use it so
    # the 8K stage shrinks at its start instead of at the end of the run.
    index_topk_start_step: int | None = None
    index_topk_end_step: int | None = None
    # Curriculum stages keep one global schedule across resumes.
    schedule_steps: int | None = None


@dataclass
class TrainConfig(BaseConfig):
    #: Optional path to a model YAML; inline ``model`` values override it.
    model_file: str | None = None
    model: ModelSettings = field(default_factory=ModelSettings)
    optim: OptimizerSettings = field(default_factory=OptimizerSettings)
    schedule: ScheduleSettings = field(default_factory=ScheduleSettings)
    data: DataSettings = field(default_factory=DataSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    distributed: DistributedSettings = field(default_factory=DistributedSettings)
    anticipatory: AnticipatoryRoutingSettings = field(
        default_factory=AnticipatoryRoutingSettings
    )


# -- reinforcement learning ---------------------------------------------------
@dataclass
class RewardSettings(BaseConfig):
    kind: str = "math"  # math | code | tool | general | world | medical
    format_bonus: float = 0.1
    parse_bonus: float = 0.1
    length_penalty: float = 0.0
    max_length_penalty_at: int = 2048
    #: MiniMind-style dense shaping.  Zero disables a term, keeping the
    #: legacy math/code reward totals unchanged.
    length_bonus: float = 0.0
    think_bonus: float = 0.0
    repetition_penalty: float = 0.0
    reward_clip: float = 0.0
    #: Optional local reward model (``internlm2-1_8b-reward`` or equivalent).
    reward_model_path: str | None = None
    reward_model_dtype: str = "float16"
    #: Tool reward weights (MiniMind ``train_agent.py`` defaults).
    tool_valid_bonus: float = 0.5
    tool_gt_weight: float = 2.5
    tool_unfinished_penalty: float = 0.5
    tool_tag_penalty: float = 0.5
    #: Code reward weights (unit tests are the verifier).
    code_timeout_s: float = 5.0
    code_pass_reward: float = 1.0
    code_compile_bonus: float = 0.0
    code_syntax_penalty: float = 0.1
    code_format_bonus: float = 0.0
    world_judge: str = "heuristic"  # heuristic | http
    world_judge_timeout_s: float = 30.0
    world_weights: dict[str, float] = field(
        default_factory=lambda: {
            "factuality": 0.35,
            "relevance": 0.20,
            "completeness": 0.15,
            "clarity": 0.15,
            "fluency": 0.10,
            "citation_support": 0.05,
        }
    )


@dataclass
class RLSettings(BaseConfig):
    method: str = "grpo"  # grpo | opd
    group_size: int = 8
    prompts_per_step: int = 4
    max_new_tokens: int = 64
    temperature: float = 1.0
    clip: float = 0.2
    #: Asymmetric PPO bounds (DAPO "clip-higher").  ``None`` keeps the symmetric
    #: ``clip``; set e.g. 0.2/0.28 only for a clip-higher ablation.
    clip_low: float | None = None
    clip_high: float | None = None
    #: MiniMind supports GRPO and CISPO; default stays GRPO for existing tests.
    loss_type: str = "grpo"  # grpo | cispo
    epsilon_high: float = 5.0
    kl_coeff: float = 0.0
    min_response_tokens: int = 2
    #: Multi-turn agent rollout (MiniMind ``train_agent.py`` defaults).
    max_turns: int = 3
    thinking_ratio: float = 0.1
    max_tool_calls: int = 4
    teacher_weights: dict[str, float] = field(default_factory=dict)
    # "torch" recomputes the full sequence (correctness reference);
    # "cached" is the batched KV-cache engine (production); "vllm" is kept for
    # environments whose vLLM build supports the model.
    rollout_backend: str = "torch"
    vllm_model_path: str | None = None
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.4
    log_domain_routing: bool = False
    weight_sync_check_every: int = 0
    weight_sync_threshold: float = 1e-3
    #: Actor-forward micro-batch size in samples.  A GRPO step packs
    #: ``prompts_per_step * group_size`` responses; forwarding all of them at
    #: once materialises O(4 x tokens x heads x window) attention activations.
    #: 0 means one batch (debug/small runs).
    ppo_mini_batch_size: int = 0
    #: Recompute the behaviour-policy log-probs with the actor before the
    #: update (verl's ``compute_log_prob``).  With a separate rollout engine the
    #: sampling log-probs differ from the actor by bf16 kernel order (~1e-2);
    #: using them directly biases the ratio.  On-policy old = actor is exact.
    recompute_old_logprobs: bool = True


@dataclass
class TeacherSettings(BaseConfig):
    name: str
    checkpoint: str | None = None
    flavor: str = "debugmodel"
    #: Optional path to a model YAML; takes precedence over ``flavor``.
    model_file: str | None = None
    seq_len: int = 512
    key: str | None = None
    weight: float = 1.0
    n_mtp_layers: int = 0
    index_topk: int | None = None
    dtype: str = "bfloat16"


@dataclass
class RLConfig(BaseConfig):
    model_file: str | None = None
    model: ModelSettings = field(default_factory=ModelSettings)
    optim: OptimizerSettings = field(default_factory=OptimizerSettings)
    schedule: ScheduleSettings = field(default_factory=ScheduleSettings)
    data: DataSettings = field(default_factory=DataSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    distributed: DistributedSettings = field(default_factory=DistributedSettings)
    rl: RLSettings = field(default_factory=RLSettings)
    reward: RewardSettings = field(default_factory=RewardSettings)
    teachers: list[TeacherSettings] = field(default_factory=list)


# -- evaluation ---------------------------------------------------------------
@dataclass
class EvalConfig(BaseConfig):
    model_file: str | None = None
    model: ModelSettings = field(default_factory=ModelSettings)
    checkpoint: str | None = None
    data_path: str | None = None
    domain: str = "math"
    max_new_tokens: int = 128
    temperature: float = 0.0
    #: Fraction of eval prompts rendered with an open thinking scaffold.
    #: Keep it equal to the RL config (general 0.9 / tool 0.1) so eval sees the
    #: same distribution the specialist was trained on.
    thinking_ratio: float = 1.0
    limit: int | None = None
    out_path: str | None = None
    reward: RewardSettings = field(default_factory=RewardSettings)


# -- helpers ------------------------------------------------------------------
def resolve_device(requested: str = "auto") -> torch.device:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def config_fingerprint(config: BaseConfig) -> str:
    import hashlib

    payload = json.dumps(config.to_dict(), sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def env_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(os.path.expandvars(os.path.expanduser(value)))
