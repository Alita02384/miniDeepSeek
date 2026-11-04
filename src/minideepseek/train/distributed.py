"""Distributed launch, FSDP2/DDP wrapping, and collective helpers.

The project targets multi-GPU CUDA (NCCL) but every helper keeps a CPU/Gloo
fallback so the same code path can be exercised by the test suite.  FSDP2 is
the composable ``fully_shard`` API (PyTorch >= 2.4): transformer blocks are
sharded individually, then the root module owns embeddings/head/norms.

Usage::

    torchrun --nproc_per_node=8 -m minideepseek.train.run --config ...
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from ..config import DistributedSettings

_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class DistributedContext:
    """Resolved distributed environment for one process."""

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    backend: str | None = None
    mesh: Any | None = None
    mode: str = "none"

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.barrier()

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(self.world_size)
        return tensor

    def broadcast_object(self, obj: Any, *, src: int = 0) -> Any:
        if not self.enabled or not dist.is_initialized():
            return obj
        objects = [obj if self.rank == src else None]
        dist.broadcast_object_list(objects, src=src)
        return objects[0]

    def shutdown(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def init_distributed(settings: DistributedSettings, requested_device: str = "auto") -> DistributedContext:
    """Initialise the process group when launched under ``torchrun``.

    ``enabled`` is inferred from ``WORLD_SIZE`` so a single-process debug run
    keeps the plain path.  Setting ``settings.enabled`` also forces init for
    tests that use ``torch.multiprocessing``.
    """
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    enabled = bool(settings.enabled or world_size > 1)

    if not enabled:
        from ..config import resolve_device

        device = resolve_device(requested_device)
        return DistributedContext(enabled=False, device=device, mode="none")

    if requested_device == "cpu":
        use_cuda = False
    elif requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()):
        use_cuda = True
    else:
        use_cuda = torch.cuda.is_available()
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("distributed run requested CUDA but torch.cuda.is_available() is False")

    backend = settings.backend
    if backend == "auto":
        backend = "nccl" if use_cuda else "gloo"
    if backend == "nccl" and not use_cuda:
        raise RuntimeError("nccl backend requires CUDA")

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30),
    )
    mesh = torch.distributed.device_mesh.init_device_mesh(
        device.type if use_cuda else "cpu", (world_size,), mesh_dim_names=("dp",)
    )
    dist.barrier()
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
        mesh=mesh,
        mode=settings.mode,
    )


def apply_parallelism(
    model: nn.Module,
    context: DistributedContext,
    settings: DistributedSettings,
    *,
    device: torch.device,
) -> nn.Module:
    """Wrap ``model`` for the configured parallelism mode.

    Returns the module that should be used for forward/backward.  FSDP2 is
    applied per transformer block + MTP block, then to the root so embeddings,
    the LM head and hc_head are sharded as one group.
    """
    if not context.enabled:
        return model
    if settings.mode == "ddp":
        # Manual data parallelism: gradients are averaged once per optimizer
        # step (after accumulation).  Keeping the raw module means the wrapper
        # API (set_index_topk / parameter_report / routing) stays unchanged.
        broadcast_buffers(model, src=0)
        return model
    if settings.mode != "fsdp2":
        raise ValueError(f"unknown distributed mode {settings.mode!r}")

    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    if settings.param_dtype not in _DTYPE_MAP:
        raise ValueError(f"unsupported param dtype {settings.param_dtype!r}")
    mp_policy = MixedPrecisionPolicy(
        param_dtype=_DTYPE_MAP[settings.param_dtype],
        reduce_dtype=_DTYPE_MAP[settings.reduce_dtype],
    )
    blocks: list[nn.Module] = []
    if hasattr(model, "layers"):
        blocks.extend(model.layers.values())
    if hasattr(model, "mtp_layers"):
        blocks.extend(model.mtp_layers)
    for block in blocks:
        fully_shard(
            block,
            mesh=context.mesh,
            mp_policy=mp_policy,
            reshard_after_forward=settings.reshard_after_forward,
        )
    fully_shard(
        model,
        mesh=context.mesh,
        mp_policy=mp_policy,
        reshard_after_forward=settings.reshard_after_forward,
    )
    broadcast_buffers(model, src=0)
    return model


def broadcast_buffers(model: nn.Module, *, src: int = 0) -> None:
    """Make hash tables / expert bias identical on every rank before training."""
    if not dist.is_initialized():
        return
    for buffer in model.buffers():
        if buffer.is_floating_point() or buffer.dtype in (
            torch.int64,
            torch.int32,
            torch.bool,
        ):
            dist.broadcast(buffer, src=src)


@contextmanager
def gradient_sync_disabled(model: nn.Module):
    """Disable FSDP2 gradient all-reduce for gradient accumulation."""
    setter = getattr(model, "set_requires_gradient_sync", None)
    if setter is None:
        yield
        return
    setter(False)
    try:
        yield
    finally:
        setter(True)


def clip_grad_norm(model: nn.Module, max_norm: float) -> float:
    """Global gradient-norm clipping that understands FSDP2 DTensor shards.

    ``torch.nn.utils.clip_grad_norm_`` computes the norm of each rank's local
    shard, which silently changes the clipping coefficient across DP sizes.
    Here sharded contributions are squared-summed locally and all-reduced, so
    the coefficient is identical to the single-process run.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    from torch.distributed.tensor import DTensor

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not parameters:
        return 0.0
    has_dtensor = any(isinstance(parameter.grad, DTensor) for parameter in parameters)
    if not has_dtensor:
        return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))

    device = parameters[0].grad.device
    # NCCL has no float64 all-reduce; float32 is sufficient for a gradient norm.
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    total = torch.zeros((), dtype=dtype, device=device)
    for parameter in parameters:
        gradient = parameter.grad
        if isinstance(gradient, DTensor):
            placements = gradient.placements
            from torch.distributed.tensor.placement_types import Replicate

            if any(isinstance(placement, Replicate) for placement in placements):
                value = gradient.full_tensor()
            else:
                value = gradient.to_local()
            total += value.detach().float().to(dtype).pow(2).sum()
        else:
            total += gradient.detach().float().to(dtype).pow(2).sum()
    if dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    norm = float(total.sqrt())
    coefficient = max_norm / (norm + 1e-6)
    if coefficient < 1.0:
        for parameter in parameters:
            parameter.grad.mul_(coefficient)
    return norm


def all_reduce_gradients(model: nn.Module, world_size: int) -> None:
    """Average gradients for the manual-DDP path."""
    if world_size <= 1:
        return
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def gather_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Full (unsharded, CPU) model state dict; collective when FSDP2 is active."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    options = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
    )
    return get_model_state_dict(model, options=options)


def _copy_full_tensor(parameter: torch.Tensor, full: torch.Tensor) -> None:
    """Copy a full CPU tensor into a parameter, sharding for DTensor."""
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Replicate, Shard

    if isinstance(parameter, DTensor):
        mesh = parameter.device_mesh
        placements = parameter.placements
        world = mesh.size()
        local_rank = mesh.get_local_rank()
        value = full.to(device=parameter.device, dtype=parameter.dtype)
        if world > 1 and isinstance(placements[0], Shard):
            chunks = torch.chunk(value, world, dim=placements[0].dim)
            value = chunks[local_rank]
        elif world > 1 and not isinstance(placements[0], Replicate):
            raise TypeError(f"unsupported placement {placements[0]!r}")
        sharded = DTensor.from_local(
            value.contiguous(), mesh, placements, shape=full.shape, stride=full.stride()
        )
        parameter.copy_(sharded)
    else:
        parameter.copy_(full.to(device=parameter.device, dtype=parameter.dtype))


@torch.no_grad()
def load_model_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a full CPU state dict into a plain/DDP/FSDP2 model.

    ``set_model_state_dict``'s broadcast path is not implemented for all
    DTensor ops on every torch build, so we copy each full tensor into its
    local shard explicitly.  This also makes resume and rollback deterministic.
    """
    named = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    missing = [name for name in named if name not in state]
    if missing:
        raise KeyError(f"checkpoint is missing parameters: {missing[:8]}")
    for name, parameter in named.items():
        _copy_full_tensor(parameter, state[name])
    for name, buffer in buffers.items():
        if name in state and torch.is_tensor(state[name]):
            buffer.copy_(state[name].to(device=buffer.device, dtype=buffer.dtype))


def gather_optimizer_state_dict(model: nn.Module, optimizer) -> dict[str, Any]:
    """Full optimizer state; collective when FSDP2 is active.

    ``HybridOptimizer`` owns a Muon and an AdamW instance; each is gathered
    separately under its own name.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
    )

    options = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
    )
    if hasattr(optimizer, "optimizers"):
        return {
            name: get_optimizer_state_dict(model, opt, options=options)
            for name, opt in zip(("muon", "adamw"), optimizer.optimizers, strict=False)
        }
    return get_optimizer_state_dict(model, optimizer, options=options)


def load_optimizer_state_dict(model: nn.Module, optimizer, state: dict[str, Any]) -> None:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_optimizer_state_dict,
    )

    options = StateDictOptions(
        full_state_dict=True, broadcast_from_rank0=True, strict=False
    )
    if hasattr(optimizer, "optimizers"):
        for name, opt in zip(("muon", "adamw"), optimizer.optimizers, strict=False):
            if name not in state:
                continue
            set_optimizer_state_dict(model, opt, state[name], options=options)
        return
    set_optimizer_state_dict(model, optimizer, state, options=options)


def is_fsdp2_module(model: nn.Module) -> bool:
    return any(hasattr(module, "_fully_shard_state") for module in model.modules())
