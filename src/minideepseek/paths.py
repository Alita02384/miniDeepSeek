"""Repository path constants.

All paths are derived from the package location so that commands work from any
working directory.  Data and checkpoints are *not* part of the git repository
and can be relocated with environment variables.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import MutableMapping
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

#: Kernel-compilation artifacts (Triton binaries, inductor output, attn_gym
#: autotune winners).  Cold compilation of the gather-attention kernels costs
#: 20-40 minutes, so the directory is repo-local and survives restarts and box
#: migrations; ``.cache/`` is git-ignored.
CACHE_DIR = Path(os.environ.get("MINIDEEPSEEK_CACHE_DIR", PROJECT_ROOT / ".cache"))


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


#: Environment variables that relocate a kernel cache, and the subdirectory of
#: :data:`CACHE_DIR` each one defaults to.
COMPILE_CACHE_ENV = {
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "ATTN_GYM_CUTE_CACHE_DIR": "attn_gym",
}


def configure_compile_cache(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Point the kernel caches at :data:`CACHE_DIR` and return the effective values.

    Triton JIT-compiles every gather-attention autotune candidate on first
    launch, and both Triton and attn_gym default their cache to ``$HOME``: a
    rebuilt instance therefore pays the full 20-40 minute compile again.
    Keeping the artifacts inside the repository makes the cost one-off; the
    same directory can be migrated with the checkpoints.

    Must run before the first ``import torch`` (Triton reads the variable when
    it is imported).  Explicit values already in the environment always win, so
    a caller can still redirect a single cache.  ``environ`` exists for tests.
    """
    env = os.environ if environ is None else environ
    root = Path(env.get("MINIDEEPSEEK_CACHE_DIR", CACHE_DIR))
    effective: dict[str, str] = {}
    for variable, default_dir in COMPILE_CACHE_ENV.items():
        value = env.setdefault(variable, str(root / default_dir))
        effective[variable] = value
        ensure_dir(Path(value))
    return effective
