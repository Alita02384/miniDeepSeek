"""Runtime compatibility helpers for the read-only torchtitan checkout.

miniDeepSeek deliberately reuses torchtitan's DeepSeek-V4 implementation instead
of forking it (see ``doc/decisions.md``).  The upstream checkout lives next to
this repository, not on PyPI, so make it importable when the environment has not
installed it.  An explicit install (``pip install -e ../torchtitan --no-deps``)
always wins; the fallback only appends the checkout to ``sys.path``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_TORCHTITAN_ENV = "MINIDEEPSEEK_TORCHTITAN_PATH"


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get(_TORCHTITAN_ENV)
    if env:
        candidates.append(Path(env).expanduser())
    # src/minideepseek/_compat.py -> repository root -> workspace root
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root.parent / "torchtitan")
    # Allow a checkout inside the repository for self-contained deployments.
    candidates.append(repo_root / "torchtitan")
    return candidates


def ensure_torchtitan() -> None:
    """Make the local torchtitan checkout importable if it is not installed."""
    if importlib.util.find_spec("torchtitan") is not None:
        return
    for candidate in _candidate_paths():
        if (candidate / "torchtitan" / "models" / "deepseek_v4").is_dir():
            sys.path.insert(0, str(candidate))
            if importlib.util.find_spec("torchtitan") is not None:
                return
    raise ModuleNotFoundError(
        "torchtitan is required but was not importable. Install the local "
        "checkout with `pip install -e ../torchtitan --no-deps` or set "
        f"{_TORCHTITAN_ENV} to its path."
    )
