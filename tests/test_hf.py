from __future__ import annotations

import pytest

pytest.importorskip("transformers")

import torch  # noqa: E402

from minideepseek.model.batch import from_sequences
from minideepseek.model.hf import MiniDeepSeekConfig, MiniDeepSeekForCausalLM
from minideepseek.model.hf.convert import hf_to_training_state, training_to_hf_state
from minideepseek.model.hf.export import export_hf


def test_hf_config_roundtrip():
    config = MiniDeepSeekConfig(
        flavor="mini_220m_a80m", seq_len=4096, max_context=8192, n_mtp_layers=1
    )
    rebuilt = MiniDeepSeekConfig.from_dict(config.to_dict())
    assert rebuilt.flavor == "mini_220m_a80m"
    assert rebuilt.seq_len == 4096
    assert rebuilt.max_context == 8192
    assert rebuilt.vocab_size == 6400
    assert rebuilt.to_model_config().n_mtp_layers == 1
    assert rebuilt.to_model_config().max_context_length == 8192


def test_state_dict_prefix_roundtrip(debug_model):
    state = debug_model.state_dict()
    roundtrip = hf_to_training_state(training_to_hf_state(state))
    assert set(roundtrip) == set(state)
    for key in state:
        assert torch.equal(roundtrip[key], state[key])


def test_hf_export_load_and_logits(tmp_path, debug_model):
    out = export_hf(
        debug_model.state_dict(),
        tmp_path / "hf",
        flavor="debugmodel",
        seq_len=64,
    )
    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists()
    assert (out / "modeling_minideepseek.py").exists()
    hf = MiniDeepSeekForCausalLM.from_pretrained(out)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        hf_logits = hf(input_ids=ids).logits
        internal = debug_model.forward(from_sequences([[1, 2, 3, 4, 5]]), use_mtp=False)[0]
    assert torch.allclose(hf_logits[0, :5], internal, atol=1e-5)


def test_hf_generation_is_deterministic(debug_model, tmp_path):
    out = export_hf(debug_model.state_dict(), tmp_path / "hf", flavor="debugmodel", seq_len=64)
    hf = MiniDeepSeekForCausalLM.from_pretrained(out)
    ids = torch.tensor([[1, 2, 3, 4]])
    first = hf.generate(ids, max_new_tokens=4, do_sample=False)
    second = hf.generate(ids, max_new_tokens=4, do_sample=False)
    assert torch.equal(first, second)
    assert first.shape == (1, 8)


def test_trust_remote_code_loading(tmp_path, monkeypatch, debug_model):
    transformers = __import__("pytest").importorskip("transformers")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    import transformers.dynamic_module_utils as dynamic

    monkeypatch.setattr(dynamic, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    out = export_hf(debug_model.state_dict(), tmp_path / "hf", flavor="debugmodel", seq_len=64)
    model = transformers.AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    assert logits.shape == (1, 3, debug_model.vocab_size)


def test_tied_embeddings_export_is_untied(tmp_path, make_debug_model):
    model = make_debug_model()
    # Force tying on the debug flavor to exercise the safetensors shared-storage path.
    model.config.tie_word_embeddings = True
    model.model.lm_head.weight = model.model.tok_embeddings.weight
    out = export_hf(
        model.state_dict(), tmp_path / "tied", flavor="debugmodel", seq_len=64
    )
    hf = MiniDeepSeekForCausalLM.from_pretrained(out)
    assert hf.model.lm_head.weight.data_ptr() != hf.model.tok_embeddings.weight.data_ptr()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        logits = hf(input_ids=ids).logits
        internal = model.forward(from_sequences([[1, 2, 3]]), use_mtp=False)[0]
    assert torch.allclose(logits[0, :3], internal, atol=1e-5)


def test_export_checkpoint_uses_meta_runtime_settings(tmp_path, debug_model):
    import json

    from minideepseek.model.checkpoint import save_checkpoint
    from minideepseek.model.hf.export import export_checkpoint

    checkpoint = tmp_path / "ckpt"
    save_checkpoint(
        checkpoint,
        debug_model,
        step=0,
        config={
            "model": {
                "flavor": "debugmodel",
                "seq_len": 64,
                "n_mtp_layers": 0,
                "swiglu_limit": 10.0,
                "index_topk": 16,
                "indexer_loss_coeff": 0.01,
            }
        },
    )
    out = export_checkpoint(
        checkpoint,
        tmp_path / "hf-meta",
        flavor="debugmodel",
        seq_len=64,
        swiglu_limit=None,
        index_topk=None,
        indexer_loss_coeff=None,
    )
    config = json.loads((out / "config.json").read_text())
    assert config["swiglu_limit"] == 10.0
    assert config["index_topk"] == 16
