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
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"top-level YAML must be a mapping: {path}")
    return data


def load_config(cls: type[T], path: str | Path) -> T:
    return _from_mapping(cls, load_yaml(path))


def dump_config(config: BaseConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True))
    return path


# -- model --------------------------------------------------------------------
@dataclass
class ModelSettings(BaseConfig):
    flavor: str = "debugmodel"
    seq_len: int = 2048
    n_mtp_layers: int = 0
    tie_word_embeddings: bool | None = None
    swiglu_limit: float | None = 10.0
    indexer_loss_coeff: float = 0.01
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
    eps: float = 1e-8
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    grad_clip: float = 1.0


@dataclass
class ScheduleSettings(BaseConfig):
    warmup_steps: int = 500
    decay: str = "cosine"  # cosine | linear | constant
    min_lr_ratio: float = 0.1


@dataclass
class DataSettings(BaseConfig):
    path: str | None = None
    tokenized_path: str | None = None
    tokenizer_dir: str | None = None
    seq_len: int | None = None
    max_records: int | None = None
    max_tokens: int | None = None
    batch_tokens: int = 4096
    shuffle: bool = True
    seed: int = 1234
    domain: str = "general"


@dataclass
class AnticipatoryRoutingSettings(BaseConfig):
    enabled: bool = False
    delta: int = 1
    max_steps: int = 100


@dataclass
class TrainSettings(BaseConfig):
    steps: int = 10
    grad_accum: int = 1
    log_every: int = 1
    ckpt_every: int = 0
    out_dir: str = "checkpoints/debug"
    mtp_weight: float = 0.3
    mtp_weight_final: float | None = None
    mtp_decay_ratio: float = 0.8
    balance_loss_coeff: float = 1e-4
    resume: str | None = None
    val_path: str | None = None
    val_every: int = 0
    val_steps: int = 10


@dataclass
class TrainConfig(BaseConfig):
    model: ModelSettings = field(default_factory=ModelSettings)
    optim: OptimizerSettings = field(default_factory=OptimizerSettings)
    schedule: ScheduleSettings = field(default_factory=ScheduleSettings)
    data: DataSettings = field(default_factory=DataSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    anticipatory: AnticipatoryRoutingSettings = field(
        default_factory=AnticipatoryRoutingSettings
    )


# -- reinforcement learning ---------------------------------------------------
@dataclass
class RewardSettings(BaseConfig):
    kind: str = "math"  # math | code | world | medical | composite
    format_bonus: float = 0.1
    parse_bonus: float = 0.1
    length_penalty: float = 0.0
    max_length_penalty_at: int = 2048
    code_timeout_s: float = 5.0
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
    kl_coeff: float = 0.0
    min_response_tokens: int = 2
    teacher_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class TeacherSettings(BaseConfig):
    name: str
    checkpoint: str | None = None
    flavor: str = "debugmodel"
    seq_len: int = 512
    key: str | None = None
    weight: float = 1.0
    n_mtp_layers: int = 0


@dataclass
class RLConfig(BaseConfig):
    model: ModelSettings = field(default_factory=ModelSettings)
    optim: OptimizerSettings = field(default_factory=OptimizerSettings)
    schedule: ScheduleSettings = field(default_factory=ScheduleSettings)
    data: DataSettings = field(default_factory=DataSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    rl: RLSettings = field(default_factory=RLSettings)
    reward: RewardSettings = field(default_factory=RewardSettings)
    teachers: list[TeacherSettings] = field(default_factory=list)


# -- evaluation ---------------------------------------------------------------
@dataclass
class EvalConfig(BaseConfig):
    model: ModelSettings = field(default_factory=ModelSettings)
    checkpoint: str | None = None
    data_path: str | None = None
    domain: str = "math"
    max_new_tokens: int = 128
    temperature: float = 0.0
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
