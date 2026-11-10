#!/usr/bin/env python3
"""Bound the ``TOPK`` unroll in attn_gym's gather-attention forward kernel.

``attn_gym/sparse/gather_attn/impl/triton/forward.py`` walks the per-query
selected slots with ``tl.static_range(0, TOPK)``, which Triton fully unrolls:
TOPK=256 therefore emits 256 copies of the online-softmax body.  On an RTX 4090
(SM89) the TTGIR pass then takes **>15 minutes per kernel and never finishes**
within a practical window, so the 8K stage (HCA width = ceil(8192/32) = 256)
could not be warmed at all, and bf16 attention was unusable there.

The same file already uses ``tl.range`` for the *backward* slot loop, and
``tl.range`` keeps the loop as an scf.for: same arithmetic, same order, but the
IR stays small.  Measured after the patch on the 8K shape (T=8192, S=256,
K=256, bf16): first call 3.0 s instead of >900 s, backward 9.9 s, steady state
3.28 ms/launch, max|diff| vs the torch reference 7.8e-3 (bf16 noise).

This is a patch to an installed dependency, not a fork: it is idempotent, keeps
a ``.orig`` backup next to the file, and ``--restore`` undoes it.  Run it once
per environment (see ``doc/server_runbook.md``); re-run after any attn_gym
upgrade, and drop it once upstream switches the forward loop to ``tl.range``.

Usage::

    .venv/bin/python scripts/patch_attn_gym.py            # patch (idempotent)
    .venv/bin/python scripts/patch_attn_gym.py --check    # exit 1 if unpatched
    .venv/bin/python scripts/patch_attn_gym.py --restore  # undo
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

OLD = "for selected_slot in tl.static_range(0, TOPK):"
#: Partial unroll: measured on the 2K shape (T=8192, S=2048, K=64, bf16) at
#: 0.566 ms/launch vs 1.061 ms fully unrolled -- a partially rolled loop is also
#: *faster* here, because 64 copies of the body thrash registers/i-cache.  At
#: K=256 it compiles in 5.8 s instead of never finishing.
UNROLL = 16
NEW = (
    f"for selected_slot in tl.range(0, TOPK, loop_unroll_factor={UNROLL}):"
    "  # patched: partial unroll (see scripts/patch_attn_gym.py)"
)
ROLLED = "for selected_slot in tl.range(0, TOPK"


def _default_target() -> pathlib.Path:
    # Resolve by path: importing the deep module needs the package's lazy
    # ``__init__`` to have run, which fails when this script is the entry point.
    import attn_gym

    root = pathlib.Path(attn_gym.__file__).parent
    return root / "sparse" / "gather_attn" / "impl" / "triton" / "forward.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=None, help="kernel file (default: installed attn_gym)")
    parser.add_argument("--check", action="store_true", help="report state, change nothing")
    parser.add_argument("--restore", action="store_true", help="undo the patch from the backup")
    args = parser.parse_args(argv)

    path = pathlib.Path(args.path) if args.path else _default_target()
    backup = path.with_suffix(".py.orig")
    text = path.read_text(encoding="utf-8")

    if args.restore:
        if not backup.exists():
            print(f"{path}: no backup at {backup}", file=sys.stderr)
            return 1
        shutil.copyfile(backup, path)
        print(f"{path}: restored from {backup}")
        return 0

    if ROLLED in text:
        if NEW in text:
            print(f"{path}: already patched (unroll={UNROLL})")
            return 0
        if args.check:
            print(f"{path}: rolled loop with a different unroll factor -- re-run without --check")
            return 1
        # A different `tl.range` variant is in place; rewrite it to the measured one.
        lines = [
            line[: len(line) - len(line.lstrip())] + NEW
            if line.strip().startswith(ROLLED)
            else line
            for line in text.splitlines()
        ]
        backup.write_text(text, encoding="utf-8")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{path}: rewrote the rolled loop to unroll={UNROLL}")
        return 0

    count = text.count(OLD)
    if count == 0:
        print(
            f"{path}: no `{OLD}` -- attn_gym changed; check whether the forward loop "
            "is already rolled and drop this script if so",
            file=sys.stderr,
        )
        return 1
    if args.check:
        print(f"{path}: {count} unrolled slot loop(s) -- needs the patch")
        return 1
    backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"{path}: patched {count} loop(s); backup at {backup}")
    print("re-run scripts/prewarm_attn.py to warm the re-keyed kernels (seconds, not hours)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
