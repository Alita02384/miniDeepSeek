"""Optional lm-evaluation-harness bridge.

The project's own ``eval/`` path is used for the domain suites.  This module
adds the roadmap's general benchmarks (MMLU/ARC/HellaSwag/C-Eval) through
``lm-eval`` when the optional extra is installed.  It loads the exported HF
directory with ``trust_remote_code=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_lm_eval() -> Any:
    try:
        import lm_eval  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "lm-evaluation-harness is not installed. Install the optional "
            "extra with `pip install -e '.[eval]'`."
        ) from exc
    return lm_eval


def run_lm_eval(
    model_path: str | Path,
    tasks: list[str],
    *,
    batch_size: int | str = 1,
    limit: int | None = None,
    num_fewshot: int | None = None,
    device: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run ``lm_eval.simple_evaluate`` against an exported HF directory."""
    lm_eval = ensure_lm_eval()
    from lm_eval.models.huggingface import HFLM

    model = HFLM(
        pretrained=str(model_path),
        trust_remote_code=True,
        batch_size=batch_size,
        device=device,
    )
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=list(tasks),
        limit=limit,
        num_fewshot=num_fewshot,
    )
    if output_path is not None:
        import json

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    return results
