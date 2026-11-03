"""Data preparation: tokenization, packing, filtering, manifests, and stats."""

from .contamination import build_eval_ngrams, find_contamination, ngram_set
from .dedup import MinHashIndex, MinHashLSH, duplicate_indices, jaccard
from .fertility import FertilityReport, fertility_report
from .grpo import GRPORecord, build_grpo_dataset, load_records, split_records, tokenize_prompts
from .io import iter_jsonl, write_jsonl
from .manifest import DataManifest, SourceSpec, load_manifest, mix_plan, validate_manifest
from .opd import OPDPrompt, balance_by_teacher, build_opd_pool, load_opd_prompts
from .pretrain import PackedPretrainDataset, build_pretrain_dataset
from .quality import QualityConfig, QualityResult, score_text
from .sft import SFTDataset, build_sft_dataset

__all__ = [
    "DataManifest",
    "FertilityReport",
    "GRPORecord",
    "MinHashIndex",
    "MinHashLSH",
    "OPDPrompt",
    "PackedPretrainDataset",
    "QualityConfig",
    "QualityResult",
    "SFTDataset",
    "SourceSpec",
    "balance_by_teacher",
    "build_eval_ngrams",
    "build_grpo_dataset",
    "build_opd_pool",
    "build_pretrain_dataset",
    "build_sft_dataset",
    "duplicate_indices",
    "fertility_report",
    "find_contamination",
    "iter_jsonl",
    "jaccard",
    "load_manifest",
    "load_opd_prompts",
    "load_records",
    "mix_plan",
    "ngram_set",
    "score_text",
    "split_records",
    "tokenize_prompts",
    "validate_manifest",
    "write_jsonl",
]
