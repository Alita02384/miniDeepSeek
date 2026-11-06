"""Hybrid Muon + AdamW optimizer.

Implements the V4 report §2.4 recipe: Muon (Newton-Schulz orthogonalization)
for hidden matrices, AdamW for embeddings, the LM head, norms, biases and router
gates.  torchtitan's ``DistMuon`` is tied to FlexShard/FSDP meshes, which are
unavailable on the CPU debug path, so this module implements the same update rule
in ~80 lines.  It is deliberately small and testable; the large-scale path can
swap in torchtitan's optimizer without changing the trainer interface.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor

from ..config import OptimizerSettings

# V4 Algorithm 1: 8 fast-convergence steps followed by 2 stabilization steps.
_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_NS_LATE_COEFFICIENTS = (2.0, -1.5, 0.5)


def zeropower_via_newtonschulz5(
    gradient: Tensor,
    *,
    steps: int = 10,
    late_steps: int = 2,
    eps: float = 1e-7,
    coefficients: tuple[float, float, float] = _NS_COEFFICIENTS,
    late_coefficients: tuple[float, float, float] = _NS_LATE_COEFFICIENTS,
) -> Tensor:
    """Hybrid Newton-Schulz orthogonalization (V4 Algorithm 1).

    ``steps - late_steps`` iterations use the fast coefficients from the
    original Muon paper; the final ``late_steps`` switch to the stabilizing
    coefficients.  ``late_steps=0`` reproduces the plain NS-5 variant.
    """
    if gradient.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {tuple(gradient.shape)}")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0 <= late_steps <= steps:
        raise ValueError("late_steps must be in [0, steps]")
    x = gradient.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for index in range(steps):
        a, b, c = (
            late_coefficients if index >= steps - late_steps else coefficients
        )
        gram = x @ x.T
        update = b * gram + c * (gram @ gram)
        x = a * x + update @ x
    if transposed:
        x = x.T
    return x.to(dtype=gradient.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer with momentum, decoupled weight decay, and NS-5."""

    def __init__(
        self,
        params: Iterable,
        *,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        ns_steps: int = 10,
        ns_late_steps: int = 2,
        update_scale: float = 0.18,
        eps: float = 1e-7,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            ns_late_steps=ns_late_steps,
            update_scale=update_scale,
            eps=eps,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim < 2:
                    raise ValueError("Muon expects parameters with at least 2 dimensions")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            ns_late_steps = int(group.get("ns_late_steps", 2))
            update_scale = float(group.get("update_scale", 0.18))
            eps = float(group["eps"])
            nesterov = bool(group["nesterov"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                full_parameter = None
                if hasattr(parameter, "full_tensor"):
                    # FSDP2 shards the parameter; orthogonalization is a global
                    # operation, so gather, update, and scatter the full matrix.
                    full_parameter = parameter.full_tensor()
                    gradient = (
                        parameter.grad.full_tensor()
                        if hasattr(parameter.grad, "full_tensor")
                        else parameter.grad
                    )
                else:
                    gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        gradient, memory_format=torch.preserve_format
                    )
                buffer = state["momentum_buffer"]
                if hasattr(buffer, "full_tensor"):
                    # A restored FSDP2 checkpoint stores the state on the
                    # parameter's sharded layout, while the Muon update is a
                    # global matrix operation on the full gradient.
                    buffer = buffer.full_tensor()
                    state["momentum_buffer"] = buffer
                buffer.mul_(momentum).add_(gradient)
                update = gradient.add(buffer, alpha=momentum) if nesterov else buffer
                matrix = update.reshape(-1, update.shape[-2], update.shape[-1])
                orthogonal = torch.stack(
                    [
                        zeropower_via_newtonschulz5(
                            slice_,
                            steps=ns_steps,
                            late_steps=ns_late_steps,
                            eps=eps,
                        )
                        for slice_ in matrix
                    ]
                ).reshape_as(update)
                rows, cols = update.shape[-2], update.shape[-1]
                scale = math.sqrt(max(rows, cols)) * update_scale
                if full_parameter is not None:
                    new_full = full_parameter.float() * (1.0 - lr * weight_decay)
                    new_full = new_full - orthogonal.float() * (lr * scale)
                    parameter.copy_(
                        _distribute_like(
                            new_full.to(full_parameter.dtype), parameter
                        )
                    )
                    continue
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(orthogonal, alpha=-lr * scale)
        return loss


class HybridOptimizer:
    """A Muon group plus an AdamW group behind one ``Optimizer``-like API."""

    def __init__(self, muon: Muon | None = None, adamw: torch.optim.AdamW | None = None):
        if muon is None and adamw is None:
            raise ValueError("at least one optimizer is required")
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self) -> list[dict]:
        groups: list[dict] = []
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [opt for opt in (self.muon, self.adamw) if opt is not None]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def state_dict(self) -> dict:
        return {
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if self.muon is not None:
            if state.get("muon") is None:
                raise ValueError("checkpoint has no Muon state")
            self.muon.load_state_dict(state["muon"])
        if self.adamw is not None:
            if state.get("adamw") is None:
                raise ValueError("checkpoint has no AdamW state")
            self.adamw.load_state_dict(state["adamw"])


def _distribute_like(full: Tensor, like: DTensor) -> DTensor:
    """Re-shard a full tensor to match an existing FSDP2 parameter."""
    mesh = like.device_mesh
    placements = like.placements
    world = mesh.size()
    local_rank = mesh.get_local_rank()
    from torch.distributed.tensor.placement_types import Replicate, Shard

    if world == 1:
        local = full
    else:
        if not isinstance(placements[0], Shard):
            if isinstance(placements[0], Replicate):
                local = full
            else:
                raise TypeError(f"unsupported placement {placements[0]!r}")
        else:
            dim = placements[0].dim
            chunks = torch.chunk(full, world, dim=dim)
            if len(chunks) != world:
                raise RuntimeError("cannot split parameter across ranks")
            local = chunks[local_rank]
    return DTensor.from_local(
        local.contiguous(),
        mesh,
        placements,
        shape=full.shape,
        stride=full.stride(),
    )


def _is_adamw_parameter(name: str, parameter: Tensor) -> bool:
    if parameter.ndim < 2:
        return True
    if "tok_embeddings" in name or "lm_head" in name:
        return True
    # Compressor positional/gate biases are static-looking vectors; the
    # reference recipe keeps them on AdamW (only wkv/wgate are Muon).
    if name.endswith(".ape"):
        return True
    # mHC static biases / gating factors use AdamW; hc_fn is a learned matrix.
    if "hc_base" in name or "hc_scale" in name:
        return True
    return parameter.numel() < 64


def build_optimizer(model, settings: OptimizerSettings) -> HybridOptimizer:
    """Split parameters into Muon and AdamW groups.

    ``settings.kind`` is ``hybrid`` (default), ``adamw`` (all AdamW) or ``muon``
    (all eligible matrices with Muon, everything else with AdamW).
    """
    muon_lr = settings.muon_lr if settings.muon_lr is not None else settings.lr
    adamw_lr = settings.adamw_lr if settings.adamw_lr is not None else settings.lr
    muon_params: list[Tensor] = []
    adamw_params: list[Tensor] = []
    muon_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        eligible = settings.kind in ("hybrid", "muon") and not _is_adamw_parameter(
            name, parameter
        )
        if eligible:
            muon_params.append(parameter)
            muon_names.append(name)
        else:
            adamw_params.append(parameter)
    if settings.kind == "adamw":
        muon_params, muon_names = [], []

    muon = None
    if muon_params:
        muon = Muon(
            [{"params": muon_params, "lr": muon_lr, "base_lr": muon_lr}],
            lr=muon_lr,
            momentum=settings.momentum,
            nesterov=settings.nesterov,
            weight_decay=settings.weight_decay,
            ns_steps=settings.ns_steps,
            ns_late_steps=settings.ns_late_steps,
            update_scale=settings.muon_update_scale,
        )
    adamw = None
    if adamw_params:
        adamw = torch.optim.AdamW(
            [{"params": adamw_params, "lr": adamw_lr, "base_lr": adamw_lr}],
            lr=adamw_lr,
            betas=settings.betas,
            eps=settings.eps,
            weight_decay=settings.weight_decay,
        )
    return HybridOptimizer(muon=muon, adamw=adamw)


__all__ = [
    "HybridOptimizer",
    "Muon",
    "build_optimizer",
    "zeropower_via_newtonschulz5",
]
