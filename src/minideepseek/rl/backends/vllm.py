"""Optional vLLM rollout backend.

The training-side log-probabilities *always* come from the torch actor; vLLM
only produces samples.  This keeps the GRPO ratio honest and lets the
consistency gate compare the two engines token by token.

Weight sync is intentionally conservative: export a fresh HF directory and
rebuild the engine, unless the installed vLLM exposes a ``reload_weights``
hook.  A production cluster should replace ``_reload`` with the weight-sync
implementation of its vLLM build.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from ...model.wrapper import GenerationOutput


def ensure_vllm() -> Any:
    try:
        import vllm  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "vLLM is not installed. Install the optional CUDA extra with "
            "`pip install -e '.[vllm]'` on a CUDA machine."
        ) from exc
    return vllm


class VLLMGenerator:
    """Batch generator backed by ``vllm.LLM``.

    Args:
        model_path: HF directory exported by ``model.hf.export``.
        tokenizer: the project tokenizer (only used for stop ids / metadata).
        tensor_parallel_size: vLLM TP degree.
        max_model_len: must cover prompt + response.
        trust_remote_code: required for the minideepseek auto_map.
    """

    def __init__(
        self,
        model_path: str | Path,
        tokenizer,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 8192,
        trust_remote_code: bool = True,
        dtype: str = "bfloat16",
        seed: int | None = None,
        flavor: str | None = None,
        swiglu_limit: float | None = 10.0,
        index_topk: int | None = None,
        indexer_loss_coeff: float = 0.01,
    ):
        self.vllm = ensure_vllm()
        self.model_path = str(model_path)
        self.tokenizer = tokenizer
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_model_len = int(max_model_len)
        self.trust_remote_code = bool(trust_remote_code)
        self.dtype = dtype
        self.seed = seed
        self.flavor = flavor
        self.swiglu_limit = swiglu_limit
        self.index_topk = index_topk
        self.indexer_loss_coeff = indexer_loss_coeff
        self._llm = None
        self.version = 0
        self._build()

    def _build(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("vLLM rollout requires CUDA")
        self._llm = self.vllm.LLM(
            model=self.model_path,
            tokenizer=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            enforce_eager=True,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        greedy: bool = False,
        stop_token_ids: Iterable[int] | None = None,
        seed: int | None = None,
    ) -> GenerationOutput:
        if not prompts:
            raise ValueError("at least one prompt is required")
        sampling = self.vllm.SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=0.0 if greedy else float(temperature),
            top_k=int(top_k) if top_k and top_k > 0 else -1,
            top_p=float(top_p),
            logprobs=0,
            seed=self.seed if seed is None else int(seed),
            stop_token_ids=list(stop_token_ids or []),
        )
        outputs = self._llm.generate(
            prompt_token_ids=[list(prompt) for prompt in prompts],
            sampling_params=sampling,
            use_tqdm=False,
        )
        sequences: list[list[int]] = []
        responses: list[list[int]] = []
        logprobs: list[list[float]] = []
        finished: list[bool] = []
        prompt_lengths = [len(prompt) for prompt in prompts]
        for prompt, result in zip(prompts, outputs, strict=True):
            completion = result.outputs[0]
            ids = list(completion.token_ids)
            sequences.append(list(prompt) + ids)
            responses.append(ids)
            logprobs.append(_extract_logprobs(completion))
            finished.append(bool(completion.finish_reason))
        return GenerationOutput(
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            response_ids=responses,
            response_logprobs=logprobs,
            finished=finished,
            temperature=1.0 if greedy else float(temperature),
        )

    def update_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        out_dir: str | Path,
        *,
        flavor: str | None = None,
    ) -> None:
        """Export new weights and refresh the engine.

        vLLM builds that expose ``reload_weights`` get an in-place refresh;
        otherwise the engine is rebuilt from the new directory.
        """
        from ...model.hf.export import export_hf

        resolved = flavor or self.flavor
        if resolved is None:
            raise ValueError("flavor is required to export vLLM weights")
        export_hf(
            state_dict,
            out_dir,
            flavor=resolved,
            seq_len=self.max_model_len,
            swiglu_limit=self.swiglu_limit,
            index_topk=self.index_topk,
            indexer_loss_coeff=self.indexer_loss_coeff,
        )
        reload = getattr(self._llm, "reload_weights", None)
        if callable(reload):
            reload(str(out_dir))
        else:
            self._llm = None
            self.model_path = str(out_dir)
            self._build()
        self.version += 1

    def close(self) -> None:
        self._llm = None


def _extract_logprobs(completion) -> list[float]:
    values: list[float] = []
    for entry in completion.logprobs or []:
        if not entry:
            values.append(float("nan"))
            continue
        # SamplingParams(logprobs=0) returns the chosen token under its id.
        chosen = max(entry.values(), key=lambda item: item.logprob)
        values.append(float(chosen.logprob))
    return values
