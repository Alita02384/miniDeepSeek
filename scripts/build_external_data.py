"""Build ``data/*.jsonl`` from the raw corpora in ``download/``.

The raw download stays untouched (and gitignored); this script applies the
domain filters from ``doc/data_plan_145m.md`` and writes the flat, upload-ready
JSONL files next to the existing MiniMind ones:

    data/pretrain_general.jsonl     fineweb-edu + chinese-fineweb-edu
                                    + cosmopedia-v2 + (sampled) finemath
    data/pretrain_code.jsonl        codeparrot-clean (license/size filtered)
    data/pretrain_tool.jsonl        ToolACE + glaive function-calling text
    data/sft_code.jsonl             code SFT in ``messages`` form
    data/grpo_code_train.jsonl      MBPP + LeetCode prompts with ``rubric.tests``
    data/eval_code.jsonl            HumanEval + HumanEval+ (evaluation only)

Every record keeps ``source``/``license`` so the manifest step can prove
provenance.  Re-running is deterministic; outputs are overwritten.

Examples::

    python scripts/build_external_data.py --only pretrain
    python scripts/build_external_data.py --only code --dry-run
    python scripts/build_external_data.py --only general --finemath-stride 7
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = ROOT / "download"
DATA = ROOT / "data"

GENERAL_JSONL = DATA / "pretrain_general.jsonl"
CODE_JSONL = DATA / "pretrain_code.jsonl"
TOOL_JSONL = DATA / "pretrain_tool.jsonl"
SFT_CODE_JSONL = DATA / "sft_code.jsonl"
GRPO_CODE_JSONL = DATA / "grpo_code_train.jsonl"
EVAL_CODE_JSONL = DATA / "eval_code.jsonl"

ALLOWED_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "isc",
    "mpl-2.0",
    "unlicense",
}
#: Pretraining code caps (doc/data_plan_145m.md §3.3).  The <=60 line /
#: <=400 token rule is the SFT口径 and is applied in ``build_sft_code``.
PRETRAIN_CODE_MAX_BYTES = 8_000
PRETRAIN_CODE_MAX_LINE = 200
PRETRAIN_CODE_MIN_ALPHA = 0.3
SFT_MAX_LINES = 60
SFT_MAX_BYTES = 8_000


def log(message: str) -> None:
    print(message, flush=True)


def parquet_rows(path: Path, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    handle = pq.ParquetFile(path)
    for batch in handle.iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


class Writer:
    """Line-buffered JSONL writer with running counters."""

    def __init__(self, path: Path, *, dry_run: bool = False) -> None:
        self.path = path
        self.dry_run = dry_run
        self.records = 0
        self.chars = 0
        self.handle = None
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any], *, text_key: str = "text") -> None:
        self.records += 1
        text = record.get(text_key)
        if isinstance(text, str):
            self.chars += len(text)
        if self.handle is not None:
            self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def report(self) -> str:
        mib = self.chars / (1024 * 1024)
        suffix = " (dry-run)" if self.dry_run else ""
        return f"{self.path.name}: {self.records} docs, {mib:.1f} M chars{suffix}"


# --------------------------------------------------------------------------- #
# pretrain
# --------------------------------------------------------------------------- #
def build_general(
    writer: Writer, *, limit: int | None, finemath_stride: int
) -> None:
    root = DOWNLOAD / "pretrain"
    if limit is not None:
        shards = sorted((root / "fineweb-edu/sample/10BT").glob("*.parquet"))
        for path in shards:
            for row in parquet_rows(path, ["text", "int_score"]):
                if (row.get("int_score") or 0) < 4:
                    continue
                text = row.get("text") or ""
                if len(text) < 200:
                    continue
                writer.write(
                    {"text": text, "domain": "general", "source": "fineweb_edu"}
                )
                if writer.records >= limit:
                    return
    else:
        for path in sorted((root / "fineweb-edu/sample/10BT").glob("*.parquet")):
            for row in parquet_rows(path, ["text", "int_score"]):
                if (row.get("int_score") or 0) < 4:
                    continue
                text = row.get("text") or ""
                if len(text) < 200:
                    continue
                writer.write(
                    {"text": text, "domain": "general", "source": "fineweb_edu"}
                )

    for path in sorted((root / "chinese-fineweb-edu/map").glob("*.parquet")):
        for row in parquet_rows(path, ["text"]):
            text = row.get("text") or ""
            if len(text) < 200:
                continue
            writer.write(
                {"text": text, "domain": "general", "source": "chinese_fineweb_edu"}
            )
            if limit is not None and writer.records >= limit:
                return

    for path in sorted((root / "cosmopedia-v2/cosmopedia-v2").glob("*.parquet")):
        for row in parquet_rows(path, ["text"]):
            text = row.get("text") or ""
            if len(text) < 200:
                continue
            writer.write(
                {"text": text, "domain": "general", "source": "cosmopedia_v2"}
            )
            if limit is not None and writer.records >= limit:
                return

    # finemath is math-heavy; keep a strided sample so "general" is not
    # dominated by it (math is no longer a specialist domain).
    stride = max(int(finemath_stride), 1)
    for path in sorted((root / "finemath-4plus/finemath-4plus").glob("*.parquet")):
        for index, row in enumerate(parquet_rows(path, ["text", "int_score"])):
            if index % stride:
                continue
            if (row.get("int_score") or 0) < 4:
                continue
            text = row.get("text") or ""
            if len(text) < 200:
                continue
            writer.write({"text": text, "domain": "general", "source": "finemath_4plus"})
            if limit is not None and writer.records >= limit:
                return


def build_pretrain_code(writer: Writer, *, limit: int | None) -> None:
    root = DOWNLOAD / "pretrain/codeparrot-clean"
    for path in sorted(root.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                license_name = str(row.get("license") or "").lower()
                if license_name not in ALLOWED_LICENSES:
                    continue
                if row.get("autogenerated"):
                    continue
                if int(row.get("line_max") or 0) > PRETRAIN_CODE_MAX_LINE:
                    continue
                if float(row.get("alpha_frac") or 0.0) < PRETRAIN_CODE_MIN_ALPHA:
                    continue
                content = row.get("content") or ""
                if not content or len(content.encode()) > PRETRAIN_CODE_MAX_BYTES:
                    continue
                writer.write(
                    {
                        "text": content,
                        "domain": "code",
                        "source": "codeparrot_clean",
                        "license": license_name,
                    }
                )
                if limit is not None and writer.records >= limit:
                    return


def _flatten_toolace(row: dict[str, Any]) -> str:
    parts = []
    system = (row.get("system") or "").strip()
    if system:
        parts.append(f"System: {system}")
    for message in row.get("conversations") or []:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        parts.append(f"{role.capitalize()}: {content}")
    return "\n".join(parts)


def build_pretrain_tool(writer: Writer, *, limit: int | None) -> None:
    toolace = DOWNLOAD / "tool/ToolACE/data.json"
    if toolace.exists():
        with toolace.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        for row in rows:
            text = _flatten_toolace(row).strip()
            if len(text) < 200:
                continue
            writer.write(
                {
                    "text": text,
                    "domain": "tool",
                    "source": "toolace",
                    "license": "apache-2.0",
                }
            )
            if limit is not None and writer.records >= limit:
                return

    glaive = DOWNLOAD / "tool/glaive-function-calling-v2/glaive-function-calling-v2.json"
    if glaive.exists():
        with glaive.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        for row in rows:
            system = (row.get("system") or "").strip()
            chat = str(row.get("chat") or "").strip()
            text = f"{system}\n{chat}".strip() if system else chat
            if len(text) < 200:
                continue
            writer.write(
                {
                    "text": text,
                    "domain": "tool",
                    "source": "glaive_function_calling",
                    "license": "apache-2.0",
                }
            )
            if limit is not None and writer.records >= limit:
                return


# --------------------------------------------------------------------------- #
# code SFT / GRPO / eval
# --------------------------------------------------------------------------- #
def _stdlib_imports_ok(tree: ast.AST) -> bool:
    stdlib = set(sys.stdlib_module_names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in stdlib:
                    return False
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] not in stdlib
        ):
            return False
    return True


def python_snippet_ok(code: str) -> bool:
    if not code or len(code.encode()) > SFT_MAX_BYTES:
        return False
    if code.count("\n") > SFT_MAX_LINES:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        return False
    return _stdlib_imports_ok(tree)


def _messages(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]


def _extract_python(code: str) -> str:
    """Keep the first ```python block when a solution is wrapped in fences."""
    if "```" not in code:
        return code
    parts = code.split("```")
    for index in range(1, len(parts), 2):
        block = parts[index]
        block = block.split("\n", 1)[1] if "\n" in block and block.split("\n", 1)[0].strip().lower() in {"python", "py"} else block
        return block.strip()
    return code


def build_sft_code(writer: Writer, *, limit: int | None) -> None:
    root = DOWNLOAD / "code/sft"

    def emit(user: str, assistant: str, source: str, license_name: str) -> bool:
        code = _extract_python(assistant)
        if not python_snippet_ok(code):
            return False
        writer.write(
            {
                "messages": _messages(user, f"```python\n{code}\n```"),
                "domain": "code",
                "source": source,
                "license": license_name,
            }
        )
        return limit is not None and writer.records >= limit

    magicoder = root / "Magicoder-OSS-Instruct-75K/data-oss_instruct-decontaminated.jsonl"
    if magicoder.exists():
        with magicoder.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("lang") != "python":
                    continue
                if emit(row.get("problem", ""), row.get("solution", ""), "magicoder_oss", "mit"):
                    return

    self_oss = root / "self-oss-instruct-sc2-exec-filter-50k/data/train-00000-of-00001.parquet"
    if self_oss.exists():
        for row in parquet_rows(self_oss, ["instruction", "response"]):
            if emit(row.get("instruction", ""), row.get("response", ""), "self_oss_instruct", "odc-by"):
                return

    codefeedback = root / "CodeFeedback-Filtered-Instruction/CodeFeedback-Filtered-Instruction.jsonl"
    if codefeedback.exists():
        with codefeedback.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if str(row.get("lang") or "").lower() != "python":
                    continue
                if emit(row.get("query", ""), row.get("answer", ""), "codefeedback", "apache-2.0"):
                    return

    python_instr = root / "python_code_instructions_18k_alpaca/data/train-00000-of-00001-8b6e212f3e1ece96.parquet"
    if python_instr.exists():
        for row in parquet_rows(python_instr, ["instruction", "input", "output"]):
            extra = (row.get("input") or "").strip()
            user = row.get("instruction", "")
            if extra:
                user = f"{user}\n\n{extra}"
            if emit(user, row.get("output", ""), "python_code_instructions_18k", "unknown"):
                return

    mbpp_full = root / "mbpp/full/train-00000-of-00001.parquet"
    if mbpp_full.exists():
        for row in parquet_rows(mbpp_full, ["text", "code"]):
            if emit(row.get("text", ""), row.get("code", ""), "mbpp", "cc-by-4.0"):
                return

    leetcode = root / "LeetCodeDataset/LeetCodeDataset-train.jsonl"
    if leetcode.exists():
        with leetcode.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if emit(
                    row.get("problem_description", ""),
                    row.get("completion", ""),
                    "leetcode",
                    "apache-2.0",
                ):
                    return


def _mbpp_tests(row: dict[str, Any]) -> str:
    setup = (row.get("test_setup_code") or "").strip()
    tests = [str(item) for item in (row.get("test_list") or [])]
    parts = [setup] if setup else []
    parts.extend(tests)
    return "\n".join(parts).strip() + "\n"


def build_grpo_code(writer: Writer, *, limit: int | None) -> None:
    root = DOWNLOAD / "code/sft"

    def emit(
        user: str,
        tests: str,
        *,
        source: str,
        license_name: str,
        reference: str = "",
        difficulty: str = "easy",
    ) -> bool:
        if not user.strip() or not tests.strip():
            return False
        writer.write(
            {
                "prompt": [{"role": "user", "content": user.strip()}],
                "domain": "code",
                "verifier": "unit_test",
                "difficulty": difficulty,
                "source": source,
                "license": license_name,
                "reference_answer": reference,
                "rubric": {"tests": tests},
            }
        )
        return limit is not None and writer.records >= limit

    mbpp_full = root / "mbpp/full/train-00000-of-00001.parquet"
    if mbpp_full.exists():
        for row in parquet_rows(mbpp_full, ["text", "code", "test_list", "test_setup_code"]):
            tests = _mbpp_tests(row)
            if emit(
                row.get("text", ""),
                tests,
                source="mbpp_full_train",
                license_name="cc-by-4.0",
                reference=row.get("code", ""),
            ):
                return

    mbpp_sanitized = root / "mbpp/sanitized/train-00000-of-00001.parquet"
    if mbpp_sanitized.exists():
        for row in parquet_rows(mbpp_sanitized, ["prompt", "code", "test_list", "test_imports"]):
            imports = "\n".join(row.get("test_imports") or [])
            tests = "\n".join([imports] + [str(item) for item in (row.get("test_list") or [])])
            if emit(
                row.get("prompt", ""),
                tests.strip() + "\n",
                source="mbpp_sanitized_train",
                license_name="cc-by-4.0",
                reference=row.get("code", ""),
            ):
                return

    leetcode = root / "LeetCodeDataset/LeetCodeDataset-train.jsonl"
    if leetcode.exists():
        with leetcode.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                test = (row.get("test") or "").strip()
                entry = (row.get("entry_point") or "").strip()
                if not test or not entry:
                    continue
                tests = f"{test}\n\ncheck({entry})\n"
                if emit(
                    row.get("problem_description", ""),
                    tests,
                    source="leetcode_train",
                    license_name="apache-2.0",
                    reference=row.get("completion", ""),
                    difficulty=str(row.get("difficulty") or "easy").lower(),
                ):
                    return


def build_eval_code(writer: Writer, *, limit: int | None) -> None:
    sources = [
        (
            DOWNLOAD / "eval/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet",
            "humaneval",
            "mit",
        ),
        (
            DOWNLOAD / "eval/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet",
            "humaneval_plus",
            "apache-2.0",
        ),
    ]
    for path, source, license_name in sources:
        if not path.exists():
            continue
        for row in parquet_rows(path, ["task_id", "prompt", "test", "entry_point"]):
            writer.write(
                {
                    "task_id": row.get("task_id"),
                    "prompt": [{"role": "user", "content": row.get("prompt", "")}],
                    "tests": f"{row.get('test', '')}\n\ncheck({row.get('entry_point')})\n",
                    "domain": "code",
                    "source": source,
                    "license": license_name,
                },
                text_key="tests",
            )
            if limit is not None and writer.records >= limit:
                return


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
TASKS = ("general", "pretrain_code", "pretrain_tool", "sft_code", "grpo_code", "eval_code")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="all",
        choices=["all", "pretrain", "code", *TASKS],
        help="which group to build",
    )
    parser.add_argument("--limit", type=int, default=None, help="max docs per output")
    parser.add_argument("--finemath-stride", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    only = args.only
    wanted = {
        "general": only in ("all", "pretrain", "general"),
        "pretrain_code": only in ("all", "pretrain", "pretrain_code"),
        "pretrain_tool": only in ("all", "pretrain", "pretrain_tool"),
        "sft_code": only in ("all", "code", "sft_code"),
        "grpo_code": only in ("all", "code", "grpo_code"),
        "eval_code": only in ("all", "code", "eval_code"),
    }

    writers: dict[str, Writer] = {}
    if wanted["general"]:
        writers["general"] = Writer(GENERAL_JSONL, dry_run=args.dry_run)
        build_general(
            writers["general"], limit=args.limit, finemath_stride=args.finemath_stride
        )
    if wanted["pretrain_code"]:
        writers["pretrain_code"] = Writer(CODE_JSONL, dry_run=args.dry_run)
        build_pretrain_code(writers["pretrain_code"], limit=args.limit)
    if wanted["pretrain_tool"]:
        writers["pretrain_tool"] = Writer(TOOL_JSONL, dry_run=args.dry_run)
        build_pretrain_tool(writers["pretrain_tool"], limit=args.limit)
    if wanted["sft_code"]:
        writers["sft_code"] = Writer(SFT_CODE_JSONL, dry_run=args.dry_run)
        build_sft_code(writers["sft_code"], limit=args.limit)
    if wanted["grpo_code"]:
        writers["grpo_code"] = Writer(GRPO_CODE_JSONL, dry_run=args.dry_run)
        build_grpo_code(writers["grpo_code"], limit=args.limit)
    if wanted["eval_code"]:
        writers["eval_code"] = Writer(EVAL_CODE_JSONL, dry_run=args.dry_run)
        build_eval_code(writers["eval_code"], limit=args.limit)

    for writer in writers.values():
        writer.close()
        log(writer.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
