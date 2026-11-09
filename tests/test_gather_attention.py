from __future__ import annotations

import pytest
import torch

from minideepseek.model.attention import gather_attention


def _attn_gym_available() -> bool:
    try:
        import attn_gym  # noqa: F401
    except Exception:
        return False
    return True


requires_fused = pytest.mark.skipif(
    not torch.cuda.is_available() or not _attn_gym_available(),
    reason="needs a CUDA device and the optional attn_gym kernels extra",
)


def _inputs(device: str, dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(0)
    batch, heads, tokens, dim = 1, 4, 256, 64
    slots, topk, window = 64, 16, 64
    q = torch.randn(batch, heads, tokens, dim, device=device, dtype=dtype)
    kv = torch.randn(batch, 1, tokens, dim, device=device, dtype=dtype)
    skv = torch.randn(batch, 1, slots, dim, device=device, dtype=dtype)
    idx = torch.randint(0, slots, (batch, tokens, topk), device=device, dtype=torch.int32)
    sink = torch.randn(heads, device=device, dtype=torch.float32)
    cu = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, slots], device=device, dtype=torch.int32)
    return q, kv, skv, idx, sink, cu, cu_k, window


def _run(impl: str, inputs, *, scale: float):
    q, kv, skv, idx, sink, cu, cu_k, window = (item for item in inputs)
    q = q.detach().clone().requires_grad_(True)
    kv = kv.detach().clone().requires_grad_(True)
    skv = skv.detach().clone().requires_grad_(True)
    sink = sink.detach().clone().requires_grad_(True)
    out, lse = gather_attention(q, kv, skv, idx, sink, cu, cu_k, window, scale=scale, impl=impl)
    out.float().sum().backward()
    return out.detach(), lse.detach(), q.grad, kv.grad, skv.grad, sink.grad


@requires_fused
def test_fused_matches_reference():
    device = "cuda"
    inputs = _inputs(device)
    scale = 0.125
    ref_out, ref_lse, ref_q, ref_kv, ref_skv, ref_sink = _run("torch", inputs, scale=scale)
    fused_out, fused_lse, fused_q, fused_kv, fused_skv, fused_sink = _run(
        "attn_gym", inputs, scale=scale
    )

    assert fused_out.shape == ref_out.shape
    torch.testing.assert_close(fused_out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fused_lse.float(), ref_lse.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(fused_q.float(), ref_q.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(fused_kv.float(), ref_kv.float(), atol=3e-2, rtol=3e-2)
    # The Triton backward accumulates sparse-KV gradients with atomics, so a
    # few bf16 elements can differ from the ordered reference by accumulation
    # order rather than by math.
    torch.testing.assert_close(fused_skv.float(), ref_skv.float(), atol=0.2, rtol=0.1)
    torch.testing.assert_close(fused_sink.float(), ref_sink.float(), atol=5e-2, rtol=5e-2)


def test_backend_torch_uses_reference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    inputs = _inputs(device, dtype=dtype)
    q, kv, skv, idx, sink, cu, cu_k, window = inputs
    out_ref, lse_ref = gather_attention(
        q, kv, skv, idx, sink, cu, cu_k, window, scale=0.125, impl="torch"
    )
    out_auto, lse_auto = gather_attention(
        q, kv, skv, idx, sink, cu, cu_k, window, scale=0.125, impl="auto"
    )
    if device == "cpu":
        torch.testing.assert_close(out_auto, out_ref)
        torch.testing.assert_close(lse_auto, lse_ref)
    else:
        # CUDA auto path may use the fused kernel; only the shapes are compared.
        assert out_auto.shape == out_ref.shape
