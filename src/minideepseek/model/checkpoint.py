"""Checkpoint save/load and safetensors export.

We keep one canonical hand-off chain (roadmap section 6)::

    training state (torch)  ->  model-only safetensors  ->  RL / eval

The training checkpoint is a directory with

``model.pt``      full ``state_dict`` (includes hash tables and expert bias)
``optimizer.pt``  optimizer state (optional)
``meta.json``     step, config, tokenizer hash, git commit, RNG state, ...

Safetensors is the portable, framework-agnostic artifact.  It is not yet an
HF-transformers architecture file (see ``doc/decisions.md``); rollout and RL use
the torch checkpoint directly.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import torch

from .. import __version__
from ..paths import repo_commit

MODEL_FILE = "model.pt"
OPTIMIZER_FILE = "optimizer.pt"
META_FILE = "meta.json"
SAFETENSORS_FILE = "model.safetensors"
SAFETENSORS_CONFIG = "config.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state().tolist(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [tensor.tolist() for tensor in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "torch" in state:
        torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if "python" in state:
        random.setstate(_tupleize(state["python"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.tensor(value, dtype=torch.uint8) for value in state["cuda"]]
        )


def _tupleize(value):
    if isinstance(value, list):
        return tuple(_tupleize(v) for v in value)
    return value


def save_checkpoint(
    directory: str | Path,
    model,
    *,
    optimizer=None,
    step: int = 0,
    config: dict[str, Any] | None = None,
    tokenizer_hash: str | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
    save_rng: bool = True,
    state_dict: dict[str, torch.Tensor] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist model (and optionally optimizer) state, returning metadata.

    ``state_dict`` / ``optimizer_state`` let a distributed caller pass already
    gathered (full, CPU) states so only rank 0 touches the filesystem.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if state_dict is None:
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    else:
        state = {key: value.detach().cpu() for key, value in state_dict.items()}
    torch.save(state, directory / MODEL_FILE)
    if optimizer is not None:
        payload = optimizer.state_dict() if optimizer_state is None else optimizer_state
        torch.save(payload, directory / OPTIMIZER_FILE)
    meta: dict[str, Any] = {
        "format": "minideepseek-torch-v1",
        "step": int(step),
        "config": config or {},
        "tokenizer_hash": tokenizer_hash,
        "seed": seed,
        "git_commit": repo_commit(),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "minideepseek_version": __version__,
        "model_sha256": _sha256(directory / MODEL_FILE),
        "extra": extra or {},
    }
    if save_rng:
        meta["rng"] = capture_rng_state()
    (directory / META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def filter_state_dict_prefixes(
    state: dict[str, torch.Tensor], prefixes: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Drop keys matching an explicit prefix (used to load main weights only).

    The V4 MTP heads are training-only; RL/eval checkpoints deliberately build
    the model with ``n_mtp_layers=0`` and must ignore those keys.  Only
    explicitly listed prefixes are dropped; all other mismatches stay errors.
    """
    if not prefixes:
        return state
    return {
        key: value
        for key, value in state.items()
        if not key.startswith(prefixes)
    }


def load_checkpoint(
    directory: str | Path,
    model,
    *,
    optimizer=None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
    ignore_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Load a checkpoint saved by :func:`save_checkpoint`."""
    directory = Path(directory)
    state = torch.load(directory / MODEL_FILE, map_location=map_location, weights_only=True)
    state = filter_state_dict_prefixes(state, ignore_prefixes)
    model.load_state_dict(state, strict=strict)
    if optimizer is not None:
        optimizer_path = directory / OPTIMIZER_FILE
        if not optimizer_path.exists():
            raise FileNotFoundError(f"optimizer state not found: {optimizer_path}")
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location=map_location, weights_only=True)
        )
    meta = read_metadata(directory)
    if meta.get("model_sha256") and _sha256(directory / MODEL_FILE) != meta["model_sha256"]:
        raise RuntimeError(f"model file checksum mismatch in {directory}")
    return meta


def read_metadata(directory: str | Path) -> dict[str, Any]:
    meta_path = Path(directory) / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"checkpoint metadata not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


# -- safetensors export -------------------------------------------------------
def export_safetensors(
    model,
    path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a portable ``model.safetensors`` plus ``config.json``."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    shared: dict[int, str] = {}
    for key, value in state.items():
        tensor = value.detach().cpu().contiguous()
        pointer = tensor.data_ptr()
        if pointer in shared:
            # safetensors rejects aliased storage.  Keep a real copy so tied
            # embeddings still satisfy a strict state-dict load on import.
            tensors[key] = tensor.clone()
            continue
        shared[pointer] = key
        tensors[key] = tensor
    save_file(tensors, str(path))
    payload = dict(config or {})
    payload.update(extra or {})
    (path.parent / SAFETENSORS_CONFIG).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def import_safetensors(path: str | Path, model, *, strict: bool = True):
    from safetensors.torch import load_file

    state = load_file(str(path))
    return model.load_state_dict(state, strict=strict)


# -- distributed checkpoint (optional) ---------------------------------------
def save_distributed_checkpoint(directory: str | Path, model, *, step: int = 0) -> Path:
    """Save a torch.distributed.checkpoint (DCP) artifact.

    DCP is the format torchtitan/verl understand natively.  A single-process
    save is supported; multi-process callers are expected to wrap this in their
    own process group.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemWriter

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    dcp.save(state, storage_writer=FileSystemWriter(str(directory)))
    (directory / "dcp_meta.json").write_text(
        json.dumps({"step": int(step), "format": "dcp"}, indent=2), encoding="utf-8"
    )
    return directory


def load_distributed_checkpoint(directory: str | Path, model) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    directory = Path(directory)
    state = {"model": {k: torch.empty_like(v) for k, v in model.state_dict().items()}}
    dcp.load(state, storage_reader=FileSystemReader(str(directory)))
    model.load_state_dict(state["model"], strict=True)


def weight_fingerprint(model) -> str:
    """Stable fingerprint used to verify rollout/actor weight sync."""
    import hashlib

    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        digest.update(name.encode())
        digest.update(param.detach().cpu().float().numpy().tobytes())
    for name, buffer in model.named_buffers():
        if torch.is_floating_point(buffer):
            digest.update(name.encode())
            digest.update(buffer.detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()
