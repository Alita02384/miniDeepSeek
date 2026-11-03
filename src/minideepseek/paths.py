"""Repository path constants.

All paths are derived from the package location so that commands work from any
working directory.  Data and checkpoints are *not* part of the git repository
and can be relocated with environment variables.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

#: Bundled copy of the MiniMind tokenizer (6,400 tokens).  Never replace this
#: silently; the tokenizer hash is part of every checkpoint's metadata.
TOKENIZER_DIR = Path(os.environ.get("MINIDEEPSEEK_TOKENIZER_DIR", PROJECT_ROOT / "tokenizer"))
TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"
TOKENIZER_CONFIG_FILE = TOKENIZER_DIR / "tokenizer_config.json"

DATA_DIR = Path(os.environ.get("MINIDEEPSEEK_DATA_DIR", PROJECT_ROOT / "data"))
CHECKPOINT_DIR = Path(os.environ.get("MINIDEEPSEEK_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
REPORT_DIR = Path(os.environ.get("MINIDEEPSEEK_REPORT_DIR", PROJECT_ROOT / "reports"))
CONFIG_DIR = PROJECT_ROOT / "configs"
DOC_DIR = PROJECT_ROOT / "doc"


def repo_commit(short: bool = True) -> str:
    """Return the current git commit, or ``"unknown"`` outside a git checkout."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out if short else out


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
