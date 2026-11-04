"""Guard the project's independence from the read-only reference checkouts."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "minideepseek"

FORBIDDEN_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+("
    r"torchtitan|minimind|LlamaFactory|llamafactory|DeepSeek_V4_Flash|deepseek_v4_flash"
    r")\b",
    re.MULTILINE,
)


def test_no_reference_checkout_imports():
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        if FORBIDDEN_IMPORT.search(text):
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, offenders


def test_no_sys_path_mutation():
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name == "_compat.py":
            offenders.append(str(path))
        text = path.read_text()
        if "sys.path.insert" in text or "sys.path.append" in text:
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, offenders


def test_package_imports_without_workspace_checkouts(tmp_path):
    import subprocess
    import sys

    code = (
        "import minideepseek\n"
        "from minideepseek.model import build_model\n"
        "m = build_model('debugmodel', seq_len=16)\n"
        "print(sum(p.numel() for p in m.parameters()))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().isdigit()
