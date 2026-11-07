"""Text artifacts must round-trip under a non-UTF-8 process locale.

The project writes JSON/YAML with ``ensure_ascii=False``/``allow_unicode=True``,
so a checkpoint or dataset created on a UTF-8 machine must still be readable on
a Windows/GBK or ``LANG=C`` host.  These tests force the C locale and assert the
production read paths still work.
"""

from __future__ import annotations

import json
import locale

import numpy as np
import pytest
from torch import nn

from minideepseek.config import TrainConfig, TrainSettings, dump_config, load_config
from minideepseek.data.io import save_flat_dataset
from minideepseek.data.pretrain import PackedPretrainDataset
from minideepseek.eval.evaluate import EvalReport, write_report
from minideepseek.model.checkpoint import read_metadata, save_checkpoint


@pytest.fixture()
def ascii_locale():
    """Switch the process text encoding to ASCII for one test."""
    try:
        previous = locale.setlocale(locale.LC_CTYPE)
        locale.setlocale(locale.LC_CTYPE, "C")
    except locale.Error:  # pragma: no cover - platform without a C locale
        pytest.skip("C locale is not available")
    try:
        if locale.getencoding().lower() not in {"us-ascii", "ascii", "ansi_x3.4-1968"}:
            pytest.skip("Python UTF-8 mode overrides the C locale")
        yield
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)


def test_checkpoint_meta_roundtrip(ascii_locale, tmp_path):
    save_checkpoint(
        tmp_path / "ckpt",
        nn.Linear(2, 2),
        step=1,
        extra={"run_name": "中文跑"},
    )
    meta = read_metadata(tmp_path / "ckpt")
    assert meta["extra"]["run_name"] == "中文跑"


def test_dataset_meta_roundtrip(ascii_locale, tmp_path):
    path = tmp_path / "toy.npz"
    save_flat_dataset(
        path,
        input_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        positions=np.array([0, 1, 2, 3], dtype=np.int32),
        target_mask=np.array([False, True, True, True]),
        offsets=np.array([0, 4], dtype=np.int64),
        meta={"sources": ["/数据/训练集.jsonl"]},
    )
    dataset = PackedPretrainDataset(path)
    assert dataset.meta["sources"] == ["/数据/训练集.jsonl"]


def test_config_roundtrip(ascii_locale, tmp_path):
    config = TrainConfig(train=TrainSettings(out_dir="/tmp/中文目录"))
    loaded = load_config(TrainConfig, dump_config(config, tmp_path / "cfg.yaml"))
    assert loaded.train.out_dir == "/tmp/中文目录"


def test_eval_report_roundtrip(ascii_locale, tmp_path):
    report = EvalReport(
        domain="math",
        samples=1,
        metrics={"accuracy": 1.0},
        examples=[{"prompt": "请证明", "response": "因为……", "reward": {"total": 1.0}}],
    )
    path = write_report(report, tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["examples"][0]["prompt"] == "请证明"
    markdown = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "请证明" in markdown
