from __future__ import annotations

import torch

from minideepseek.model import build_model, from_sequences
from minideepseek.model.checkpoint import (
    export_safetensors,
    import_safetensors,
    load_checkpoint,
    save_checkpoint,
    weight_fingerprint,
)


def test_torch_checkpoint_roundtrip(debug_model, tmp_path):
    checkpoint = tmp_path / "ckpt"
    save_checkpoint(checkpoint, debug_model, step=7, tokenizer_hash="abc")
    reloaded = build_model("debugmodel", seq_len=128, n_mtp_layers=0)
    meta = load_checkpoint(checkpoint, reloaded)
    assert meta["step"] == 7
    assert meta["tokenizer_hash"] == "abc"
    batch = from_sequences([[1, 2, 3, 4]])
    with torch.no_grad():
        original = debug_model.forward(batch)[0]
        restored = reloaded.forward(batch)[0]
    assert torch.equal(original, restored)


def test_safetensors_roundtrip(debug_model, tmp_path):
    path = tmp_path / "model.safetensors"
    export_safetensors(debug_model, path, config={"flavor": "debugmodel"})
    assert path.exists()
    assert (tmp_path / "config.json").exists()
    reloaded = build_model("debugmodel", seq_len=128, n_mtp_layers=0)
    import_safetensors(path, reloaded)
    batch = from_sequences([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.equal(debug_model.forward(batch)[0], reloaded.forward(batch)[0])


def test_weight_fingerprint_changes(debug_model):
    before = weight_fingerprint(debug_model)
    with torch.no_grad():
        next(debug_model.parameters()).add_(1.0)
    after = weight_fingerprint(debug_model)
    assert before != after


def test_checkpoint_rng_state(debug_model, tmp_path):
    torch.manual_seed(123)
    checkpoint = tmp_path / "ckpt"
    save_checkpoint(checkpoint, debug_model)
    expected = torch.rand(4)
    meta = load_checkpoint(checkpoint, debug_model)
    assert "rng" in meta
    from minideepseek.model.checkpoint import restore_rng_state

    restore_rng_state(meta["rng"])
    assert torch.allclose(torch.rand(4), expected)


def test_distributed_checkpoint_roundtrip(debug_model, tmp_path):
    from minideepseek.model.checkpoint import (
        load_distributed_checkpoint,
        save_distributed_checkpoint,
    )

    directory = tmp_path / "dcp"
    save_distributed_checkpoint(directory, debug_model, step=3)
    reloaded = build_model("debugmodel", seq_len=128, n_mtp_layers=0)
    load_distributed_checkpoint(directory, reloaded)
    batch = from_sequences([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.equal(debug_model.forward(batch)[0], reloaded.forward(batch)[0])
