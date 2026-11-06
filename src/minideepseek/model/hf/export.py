"""Export a training checkpoint to a Hugging Face directory.

The exported directory contains ``config.json`` (with ``auto_map``),
``model.safetensors``, and copies of the two remote-code files, so both
``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`` and a
plain ``MiniDeepSeekForCausalLM.from_pretrained`` work.

MTP layers are training-only auxiliary heads; they are dropped from the export
so the HF model matches what inference engines expect.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from minideepseek.model.hf.configuration_minideepseek import MiniDeepSeekConfig
from minideepseek.model.hf.convert import training_to_hf_state
from minideepseek.model.hf.modeling_minideepseek import MiniDeepSeekForCausalLM


def _code_files() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        here / "configuration_minideepseek.py",
        here / "modeling_minideepseek.py",
    ]


def export_hf(
    state_dict: dict[str, torch.Tensor],
    out_dir: str | Path,
    *,
    flavor: str,
    seq_len: int = 8192,
    max_context: int | None = None,
    tie_word_embeddings: bool | None = None,
    swiglu_limit: float | None = 10.0,
    index_topk: int | None = None,
    indexer_loss_coeff: float = 0.01,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Export untied: safetensors rejects aliased storages, and inference
    # engines prefer a plain ``lm_head.weight`` over a shared tensor.  The
    # training checkpoint contains both keys, so logits stay identical.
    del tie_word_embeddings
    config = MiniDeepSeekConfig(
        flavor=flavor,
        seq_len=seq_len,
        max_context=max_context,
        n_mtp_layers=0,
        tie_word_embeddings=False,
        swiglu_limit=swiglu_limit,
        index_topk=index_topk,
        indexer_loss_coeff=indexer_loss_coeff,
    )
    model = MiniDeepSeekForCausalLM(config)
    main_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("mtp_layers.")
    }
    incompatible = model.load_state_dict(training_to_hf_state(main_state), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected keys for HF export: {incompatible.unexpected_keys}")
    missing = [key for key in incompatible.missing_keys if not key.startswith("model.mtp_layers.")]
    if missing:
        raise RuntimeError(f"missing keys for HF export: {missing}")
    model.save_pretrained(out_dir, safe_serialization=True)

    # save_pretrained only copies remote code when the class was registered;
    # copy explicitly so the directory is self-describing.
    for path in _code_files():
        shutil.copy2(path, out_dir / path.name)

    # The rollout engine needs the exact project tokenizer (6,400 vocab +
    # MiniMind chat template), not a generic one.
    from minideepseek.paths import TOKENIZER_DIR

    tokenizer_dir = TOKENIZER_DIR
    for name in ("tokenizer.json", "tokenizer_config.json"):
        source = tokenizer_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
    payload = json.loads((out_dir / "config.json").read_text())
    payload["auto_map"] = {
        "AutoConfig": "configuration_minideepseek.MiniDeepSeekConfig",
        "AutoModelForCausalLM": "modeling_minideepseek.MiniDeepSeekForCausalLM",
    }
    payload["architectures"] = ["MiniDeepSeekForCausalLM"]
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
    return out_dir


def export_checkpoint(checkpoint: str | Path, out_dir: str | Path, **kwargs) -> Path:
    """Export ``<checkpoint>/model.pt`` written by ``model.checkpoint``."""
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    meta = json.loads((checkpoint / "meta.json").read_text())
    config = meta.get("config", {})
    model_cfg = config.get("model", {})
    if kwargs.get("flavor") is None:
        kwargs["flavor"] = model_cfg.get("flavor")
    if kwargs.get("seq_len") is None:
        kwargs["seq_len"] = model_cfg.get("seq_len")
    if kwargs.get("max_context") is None:
        kwargs["max_context"] = model_cfg.get("max_context")
    if kwargs.get("swiglu_limit") is None:
        kwargs["swiglu_limit"] = model_cfg.get("swiglu_limit", 10.0)
    if kwargs.get("index_topk") is None:
        # The trainer records the *runtime* top-k (the static config may hold
        # the pre-schedule value).  Prefer it when present.
        runtime_topk = meta.get("extra", {}).get("index_topk")
        kwargs["index_topk"] = (
            runtime_topk if runtime_topk is not None else model_cfg.get("index_topk")
        )
    if kwargs.get("indexer_loss_coeff") is None:
        kwargs["indexer_loss_coeff"] = model_cfg.get("indexer_loss_coeff", 0.01)
    if kwargs["flavor"] is None or kwargs["seq_len"] is None:
        raise ValueError(
            "checkpoint meta lacks model.flavor/seq_len; pass them explicitly"
        )
    return export_hf(state, out_dir, **kwargs)
