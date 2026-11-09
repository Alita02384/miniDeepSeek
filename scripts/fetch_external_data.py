#!/usr/bin/env python3
"""Fetch the external corpora required by ``doc/data_plan_145m.md``.

The 145M run needs ~3B pretraining tokens and a code SFT/GRPO set.  Pulling the
upstream corpora in full would cost hundreds of GB, so each source is cut down
to a few shards: roughly 2-3x the planned token budget per group, leaving room
for the quality filters (``doc/data_plan_145m.md`` §3.3, §4.1).  Shard counts and
the reasoning are recorded in ``doc/external_data.md``.

Everything lands under ``download/<group>/<dataset>/`` in the same layout the HF
repo uses, together with ``download/manifest.json`` (bytes + sha256 + revision).

Usage::

    python scripts/fetch_external_data.py --jobs 3
    python scripts/fetch_external_data.py --dry-run
    python scripts/fetch_external_data.py --only code
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = ROOT / "download"

# Keep the HF cache inside the repo (writable in the sandbox) and avoid xet,
# which was failing with SSL EOF on this network.
os.environ.setdefault("HF_HOME", str(DOWNLOAD / ".hf_home"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@dataclass(frozen=True)
class Dataset:
    group: str
    repo: str
    dest: str
    files: tuple[str, ...]
    license: str
    note: str


def _files(count: int, pattern: str, start: int = 0) -> tuple[str, ...]:
    return tuple(pattern.format(i=i) for i in range(start, start + count))


PLAN: tuple[Dataset, ...] = (
    # ---- pretraining (§3.2) -------------------------------------------------
    Dataset(
        group="pretrain",
        repo="HuggingFaceFW/fineweb-edu",
        dest="pretrain/fineweb-edu",
        files=_files(4, "sample/10BT/{i:03d}_00000.parquet"),
        license="ODC-By",
        note="English edu web; int_score>=4 keeps ~13%, 1 shard ~= 160M kept tokens, need 450M",
    ),
    Dataset(
        group="pretrain",
        repo="opencsg/chinese-fineweb-edu",
        dest="pretrain/chinese-fineweb-edu",
        files=_files(1, "map/{i:05d}.parquet", start=22),
        license="Apache-2.0",
        note="Chinese edu web (MAP-CC slice); 1 x 2.6GB ~= 1B tokens, need 450M",
    ),
    Dataset(
        group="pretrain",
        repo="codeparrot/codeparrot-clean",
        dest="pretrain/codeparrot-clean",
        files=_files(16, "file-{i:012d}.json.gz", start=1),
        license="permissive (per-file license field)",
        note="Python source; <=200 lines + permissive keeps ~34%, 16 shards ~= 750M tokens",
    ),
    Dataset(
        group="pretrain",
        repo="HuggingFaceTB/smollm-corpus",
        dest="pretrain/cosmopedia-v2",
        files=_files(2, "cosmopedia-v2/train-{i:05d}-of-00104.parquet"),
        license="ODC-By",
        note="Synthetic textbooks; 1 shard ~= 500M tokens, need 300M",
    ),
    Dataset(
        group="pretrain",
        repo="HuggingFaceTB/finemath",
        dest="pretrain/finemath-4plus",
        files=_files(2, "finemath-4plus/train-{i:05d}-of-00064.parquet"),
        license="ODC-By",
        note="Math, int_score>=4; 2 x 286MB, need 60M tokens",
    ),
    # ---- tool (§3.2 tool 3%) ------------------------------------------------
    # Salesforce/xlam-function-calling-60k is gated now (401 without a token);
    # ToolACE + glaive + the local data/sft_tool.jsonl cover the schema pool.
    Dataset(
        group="tool",
        repo="Team-ACE/ToolACE",
        dest="tool/ToolACE",
        files=("data.json",),
        license="Apache-2.0",
        note="11.3k tool SFT; schema diversity for GRPO prompts",
    ),
    Dataset(
        group="tool",
        repo="glaiveai/glaive-function-calling-v2",
        dest="tool/glaive-function-calling-v2",
        files=("glaive-function-calling-v2.json",),
        license="Apache-2.0",
        note="112k single/double call dialogues",
    ),
    # ---- code SFT (§4.1) ----------------------------------------------------
    Dataset(
        group="code",
        repo="google-research-datasets/mbpp",
        dest="code/sft/mbpp",
        files=(
            "full/train-00000-of-00001.parquet",
            "full/validation-00000-of-00001.parquet",
            "full/test-00000-of-00001.parquet",
            "full/prompt-00000-of-00001.parquet",
            "sanitized/train-00000-of-00001.parquet",
            "sanitized/validation-00000-of-00001.parquet",
            "sanitized/test-00000-of-00001.parquet",
        ),
        license="CC-BY-4.0",
        note="function-level anchor; train only for fitting, test is eval",
    ),
    Dataset(
        group="code",
        repo="ise-uiuc/Magicoder-OSS-Instruct-75K",
        dest="code/sft/Magicoder-OSS-Instruct-75K",
        files=("data-oss_instruct-decontaminated.jsonl",),
        license="MIT",
        note="75k OSS-instruct; keep lang==python and short solutions",
    ),
    Dataset(
        group="code",
        repo="bigcode/self-oss-instruct-sc2-exec-filter-50k",
        dest="code/sft/self-oss-instruct-sc2-exec-filter-50k",
        files=("data/train-00000-of-00001.parquet",),
        license="ODC-By",
        note="50k execution-filtered, most reliable code SFT subset",
    ),
    Dataset(
        group="code",
        repo="m-a-p/CodeFeedback-Filtered-Instruction",
        dest="code/sft/CodeFeedback-Filtered-Instruction",
        files=("CodeFeedback-Filtered-Instruction.jsonl",),
        license="Apache-2.0",
        note="156k feedback-filtered; take the simple python subset only",
    ),
    Dataset(
        group="code",
        repo="iamtarun/python_code_instructions_18k_alpaca",
        dest="code/sft/python_code_instructions_18k_alpaca",
        files=("data/train-00000-of-00001-8b6e212f3e1ece96.parquet",),
        license="check upstream (alpaca-derived)",
        note="18k python instructions; keep def + docstring rows",
    ),
    Dataset(
        group="code",
        repo="newfacade/LeetCodeDataset",
        dest="code/sft/LeetCodeDataset",
        files=("LeetCodeDataset-train.jsonl", "LeetCodeDataset-test.jsonl"),
        license="check upstream",
        note="2859 LeetCode problems with tests; closest to the target skill",
    ),
    # ---- code GRPO (§4.2) ---------------------------------------------------
    Dataset(
        group="code",
        repo="open-r1/verifiable-coding-problems-python_decontaminated-tested",
        dest="code/grpo/open-r1-verifiable",
        files=_files(3, "data/train-{i:05d}-of-00003.parquet"),
        license="check upstream",
        note="15k verified problems with fn_name/input/output",
    ),
    Dataset(
        group="code",
        repo="BAAI/TACO",
        dest="code/grpo/TACO",
        files=_files(4, "train/data-{i:05d}-of-00009.arrow") + ("test/data-00000-of-00001.arrow",),
        license="Apache-2.0",
        note="competition problems; 4/9 train shards is enough EASY pool (~3k), test is eval",
    ),
    # ---- eval (§8) ----------------------------------------------------------
    Dataset(
        group="eval",
        repo="openai/openai_humaneval",
        dest="eval/openai_humaneval",
        files=("openai_humaneval/test-00000-of-00001.parquet",),
        license="MIT",
        note="HumanEval; eval only, never train",
    ),
    Dataset(
        group="eval",
        repo="evalplus/humanevalplus",
        dest="eval/humanevalplus",
        files=("data/test-00000-of-00001-5973903632b82d40.parquet",),
        license="Apache-2.0",
        note="HumanEval+; eval only, never train",
    ),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(ds: Dataset, rel: str, tries: int = 8) -> tuple[str, str, int]:
    from huggingface_hub import hf_hub_download

    target = DOWNLOAD / ds.dest / rel
    if target.is_file() and target.stat().st_size > 0:
        return rel, "cached", target.stat().st_size

    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            hf_hub_download(
                repo_id=ds.repo,
                filename=rel,
                repo_type="dataset",
                local_dir=str(DOWNLOAD / ds.dest),
            )
            return rel, "ok", target.stat().st_size
        except Exception as exc:  # noqa: BLE001 - network errors are broad
            last = exc
            if attempt == tries:
                break
            time.sleep(min(2**attempt, 60))
    raise RuntimeError(f"{ds.repo}:{rel} failed after {tries} tries: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=3, help="parallel file downloads")
    ap.add_argument("--only", action="append", choices=["pretrain", "tool", "code", "eval"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 in manifest")
    args = ap.parse_args()

    plan = [ds for ds in PLAN if not args.only or ds.group in args.only]
    tasks = [(ds, rel) for ds in plan for rel in ds.files]
    total = len(tasks)

    if args.dry_run:
        print(f"{total} files from {len(plan)} datasets -> {DOWNLOAD}")
        for ds in plan:
            print(f"  [{ds.group}] {ds.repo} -> {ds.dest} ({len(ds.files)} files)")
        return 0

    DOWNLOAD.mkdir(parents=True, exist_ok=True)
    print(f"downloading {total} files from {len(plan)} datasets into {DOWNLOAD}")
    started = time.time()
    done = 0
    failures: list[str] = []
    bytes_done = 0

    # Code SFT first (small, unblocks the next pipeline step), pretrain last.
    order = {"code": 0, "tool": 1, "eval": 2, "pretrain": 3}
    queue = [(ds, rel) for ds in sorted(plan, key=lambda d: order[d.group]) for rel in ds.files]

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_download_one, ds, rel): (ds, rel) for ds, rel in queue}
        for fut in as_completed(futures):
            ds, rel = futures[fut]
            done += 1
            try:
                _, status, size = fut.result()
                bytes_done += size
                elapsed = time.time() - started
                print(
                    f"[{done:3d}/{total}] {status:6s} {size / 1e9:7.3f} GB  "
                    f"{ds.repo}:{rel}  ({bytes_done / 1e9:.1f} GB, {elapsed / 60:.1f} min)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{ds.repo}:{rel}: {exc}")
                print(f"[{done:3d}/{total}] FAILED {ds.repo}:{rel}: {exc}", flush=True)

    # ---- manifest -----------------------------------------------------------
    from huggingface_hub import HfApi

    api = HfApi()
    revisions: dict[str, str] = {}
    for ds in plan:
        try:
            revisions[ds.repo] = api.dataset_info(ds.repo).sha or ""
        except Exception:  # noqa: BLE001 - offline manifest is still useful
            revisions[ds.repo] = ""

    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "hf_home": str(DOWNLOAD / ".hf_home"),
        "datasets": [],
    }
    entries = []
    for ds in plan:
        files = []
        for rel in ds.files:
            path = DOWNLOAD / ds.dest / rel
            if not path.is_file():
                continue
            entry = {"path": rel, "bytes": path.stat().st_size}
            if not args.no_hash:
                entry["sha256"] = _sha256(path)
            files.append(entry)
        entries.append(
            {
                "group": ds.group,
                "repo": ds.repo,
                "revision": revisions.get(ds.repo, ""),
                "dest": ds.dest,
                "license": ds.license,
                "note": ds.note,
                "files": files,
            }
        )
    manifest["datasets"] = entries
    out = DOWNLOAD / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nmanifest -> {out}")

    if failures:
        print(f"\n{len(failures)} file(s) failed; rerun to resume:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"done in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
