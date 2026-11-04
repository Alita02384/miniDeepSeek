"""Optional verl adapter: data format, reward entrypoint, and Hydra config.

The canonical GRPO/OPD implementation in this repository is ``rl/grpo.py`` /
``rl/opd.py``.  This module makes the same data and rewards usable by verl's
FSDP actor with the HF rollout (``rollout.name=hf``) or vLLM when the installed
vLLM build supports the model.  ``verl`` itself is an optional extra; importing
this module never imports it.

verl expects parquet datasets with these columns::

    prompt: list[{"role": ..., "content": ...}]
    data_source: str
    reward_model: {"ground_truth": str, "style": "rule"}
    extra_info: dict

Use ``write_verl_dataset(records, path)`` to produce JSONL (always) and parquet
(when pandas/pyarrow are installed).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from ..config import RewardSettings
from ..data.grpo import GRPORecord
from .reward import dispatch_reward

_REWARD_FILE = str(Path(__file__).resolve())


def to_verl_record(record: GRPORecord, *, data_source: str | None = None) -> dict[str, Any]:
    """Convert one internal GRPO record to verl's dataset schema."""
    source = data_source or f"minideepseek/{record.domain}/{record.verifier}"
    return {
        "prompt": record.prompt,
        "data_source": source,
        "reward_model": {
            "ground_truth": record.reference_answer or "",
            "style": "rule",
        },
        "extra_info": {
            "domain": record.domain,
            "verifier": record.verifier,
            "rubric": record.rubric,
            "difficulty": record.difficulty,
            "source": record.source,
            "license": record.license,
            "teacher_key": record.domain,
        },
        "ability": record.domain,
    }


def write_verl_dataset(records: Sequence[GRPORecord], path: str | Path) -> Path:
    """Write JSONL; also write parquet next to it when pyarrow is available."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [to_verl_record(record) for record in records]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    try:
        import pandas as pd
    except ImportError:
        return path
    parquet = path.with_suffix(".parquet")
    pd.DataFrame(rows).to_parquet(parquet)
    return parquet


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """verl ``custom_reward_function`` entrypoint.

    ``data_source`` is ``minideepseek/<domain>/<verifier>``; ``extra_info``
    carries the rubric used by code/world rewards.
    """
    extra = extra_info or {}
    domain = str(extra.get("domain") or "general")
    verifier = str(extra.get("verifier") or "none")
    if data_source.startswith("minideepseek/"):
        parts = data_source.split("/")
        if len(parts) == 3:
            domain, verifier = parts[1], parts[2]
    record = GRPORecord(
        prompt=[],
        domain=domain,
        reference_answer=ground_truth or None,
        rubric=dict(extra.get("rubric") or {}),
        verifier=verifier,
        difficulty=str(extra.get("difficulty") or "medium"),
        source=str(extra.get("source") or "verl"),
        license=str(extra.get("license") or "unknown"),
    )
    settings = RewardSettings(kind=domain)
    return float(dispatch_reward(solution_str, record, settings).total)


def build_verl_config(
    *,
    hf_model_path: str | Path,
    train_files: list[str],
    val_files: list[str],
    output_dir: str,
    n_gpus_per_node: int = 8,
    nnodes: int = 1,
    total_epochs: int = 10,
    train_batch_size: int = 128,
    ppo_mini_batch_size: int = 32,
    max_prompt_length: int = 1024,
    max_response_length: int = 1024,
    actor_lr: float = 1e-6,
    rollout: str = "hf",
    rollout_tp: int = 1,
    gpu_memory_utilization: float = 0.4,
    project_name: str = "minideepseek",
    experiment_name: str = "grpo",
    save_freq: int = 100,
    test_freq: int = 10,
) -> dict[str, Any]:
    """Hydra config for ``verl.trainer.main_ppo`` (GRPO, FSDP actor)."""
    if rollout not in ("hf", "vllm"):
        raise ValueError("rollout must be 'hf' or 'vllm'")
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_files": train_files,
            "val_files": val_files,
            "train_batch_size": int(train_batch_size),
            "max_prompt_length": int(max_prompt_length),
            "max_response_length": int(max_response_length),
            "filter_overlong_prompts": True,
            "truncation": "error",
            "shuffle": True,
            "return_raw_chat": True,
        },
        "actor_rollout_ref": {
            "model": {
                "path": str(hf_model_path),
                "trust_remote_code": True,
                "use_remove_padding": False,
                "enable_gradient_checkpointing": False,
            },
            "actor": {
                "strategy": "fsdp",
                "use_torch_compile": False,
                "optim": {"lr": float(actor_lr)},
                "ppo_mini_batch_size": int(ppo_mini_batch_size),
                "fsdp_config": {
                    "param_offload": False,
                    "optimizer_offload": False,
                },
            },
            "rollout": {
                "name": rollout,
                "mode": "sync",
                "n": 1,
                "tensor_model_parallel_size": int(rollout_tp),
                "gpu_memory_utilization": float(gpu_memory_utilization),
                "max_model_len": int(max_prompt_length + max_response_length),
                "log_prob_micro_batch_size_per_gpu": 4,
            },
        },
        "custom_reward_function": {
            "path": _REWARD_FILE,
            "name": "compute_score",
        },
        "trainer": {
            "balance_batch": True,
            "logger": ["console"],
            "project_name": project_name,
            "experiment_name": experiment_name,
            "n_gpus_per_node": int(n_gpus_per_node),
            "nnodes": int(nnodes),
            "val_before_train": False,
            "total_epochs": int(total_epochs),
            "save_freq": int(save_freq),
            "test_freq": int(test_freq),
            "default_local_dir": str(output_dir),
        },
    }


def flatten_overrides(config: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten a nested config into Hydra ``key=value`` overrides."""
    items: list[str] = []
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(flatten_overrides(value, name))
        elif isinstance(value, list):
            items.append(f"{name}={json.dumps(value, ensure_ascii=False)}")
        elif isinstance(value, bool):
            items.append(f"{name}={'true' if value else 'false'}")
        else:
            items.append(f"{name}={value}")
    return items


def write_verl_config(config: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def build_verl_opd_config(
    *,
    hf_student_path: str | Path,
    teachers: dict[str, dict[str, Any]],
    train_files: list[str],
    val_files: list[str],
    output_dir: str,
    n_gpus_per_node: int = 8,
    total_epochs: int = 10,
    loss_mode: str = "forward_kl_topk",
    topk: int = 1024,
    **kwargs: Any,
) -> dict[str, Any]:
    """verl OPD template (multi-teacher, FSDP).

    ``teachers`` maps a data-source key to ``{"model_path": ..., "num_replicas":
    ...}``.  The V4-faithful full-vocabulary reverse KL is implemented by
    ``rl/opd.py``; verl's ``forward_kl_topk`` is the closest available mode and
    should use ``topk`` = vocabulary size when memory permits.
    """
    config = build_verl_config(
        hf_model_path=hf_student_path,
        train_files=train_files,
        val_files=val_files,
        output_dir=output_dir,
        n_gpus_per_node=n_gpus_per_node,
        total_epochs=total_epochs,
        **kwargs,
    )
    distillation: dict[str, Any] = {
        "enabled": True,
        "n_gpus_per_node": int(n_gpus_per_node),
        "nnodes": 1,
        "teacher_key": "data_source",
        "teacher_models": {
            key: {
                "key": key,
                "model_path": str(value["model_path"]),
                "num_replicas": int(value.get("num_replicas", 1)),
            }
            for key, value in teachers.items()
        },
        "distillation_loss": {
            "loss_mode": loss_mode,
            "topk": int(topk),
            "use_policy_gradient": False,
            "distillation_loss_coef": 1.0,
        },
    }
    config["distillation"] = distillation
    return config
