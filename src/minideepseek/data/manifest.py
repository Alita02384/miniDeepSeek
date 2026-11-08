"""Dataset manifests: mixing weights, token budgets, and epoch limits.

A manifest is the contract between data preparation and pretraining.  It records
where every token came from, its license, and how many epochs it may be reused
(roadmap sections 4.1, 4.2 and 4.7).  ``validate`` fails loudly on the two
mistakes that matter most: weights that do not sum to one, and a single source
being reused more than four epochs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import BaseConfig, load_config
from .io import jsonl_fingerprint

DOMAINS = {"general", "world", "code", "math", "medical", "agent", "tool", "mixed"}
MAX_EPOCHS = 4


@dataclass
class SourceSpec(BaseConfig):
    name: str
    path: str
    domain: str = "general"
    license: str = "unknown"
    weight: float = 0.0
    available_tokens: int | None = None
    max_epochs: float = MAX_EPOCHS


@dataclass
class DataManifest(BaseConfig):
    name: str = "unnamed"
    seq_len: int = 4096
    total_tokens: int = 0
    sources: list[SourceSpec] = field(default_factory=list)


@dataclass
class MixEntry:
    name: str
    domain: str
    license: str
    target_tokens: int
    available_tokens: int | None
    epochs: float | None


@dataclass
class ManifestReport(BaseConfig):
    name: str
    total_tokens: int
    seq_len: int
    entries: list[MixEntry]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _fingerprint(path: str) -> str | None:
    candidate = Path(path)
    if candidate.is_file():
        return jsonl_fingerprint(candidate)
    # Packed datasets keep the arrays in ``.npy`` sidecars and only a JSON meta
    # next to the logical ``.npz`` path; fingerprint the meta in that case.
    meta = candidate.with_suffix(".json")
    if meta.is_file():
        return jsonl_fingerprint(meta)
    return None


def load_manifest(path: str | Path) -> DataManifest:
    return load_config(DataManifest, path)


def validate_manifest(manifest: DataManifest) -> ManifestReport:
    errors: list[str] = []
    warnings: list[str] = []
    if manifest.total_tokens <= 0:
        errors.append("total_tokens must be positive")
    if manifest.seq_len <= 0:
        errors.append("seq_len must be positive")
    if not manifest.sources:
        errors.append("manifest has no sources")
    names = [source.name for source in manifest.sources]
    if len(set(names)) != len(names):
        errors.append("source names must be unique")
    weight_sum = sum(source.weight for source in manifest.sources)
    if manifest.sources and not math.isclose(weight_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        errors.append(f"weights must sum to 1.0, got {weight_sum:.6f}")
    entries: list[MixEntry] = []
    for source in manifest.sources:
        if source.domain not in DOMAINS:
            errors.append(f"{source.name}: unknown domain {source.domain!r}")
        if source.weight < 0:
            errors.append(f"{source.name}: negative weight")
        if source.max_epochs > MAX_EPOCHS:
            errors.append(
                f"{source.name}: max_epochs={source.max_epochs} exceeds the "
                f"{MAX_EPOCHS}-epoch cap"
            )
        target = int(round(source.weight * manifest.total_tokens))
        epochs = None
        if source.available_tokens:
            epochs = target / source.available_tokens
            if epochs > min(source.max_epochs, MAX_EPOCHS) + 1e-9:
                errors.append(
                    f"{source.name}: target {target} tokens would repeat the source "
                    f"{epochs:.2f} epochs (> {min(source.max_epochs, MAX_EPOCHS)})"
                )
        elif target > 0:
            warnings.append(
                f"{source.name}: available_tokens unknown; cannot verify the epoch cap"
            )
        fingerprint = _fingerprint(source.path)
        if fingerprint is None:
            warnings.append(f"{source.name}: path not found ({source.path})")
        entries.append(
            MixEntry(
                name=source.name,
                domain=source.domain,
                license=source.license,
                target_tokens=target,
                available_tokens=source.available_tokens,
                epochs=epochs,
            )
        )
    if entries:
        assigned = sum(entry.target_tokens for entry in entries)
        if assigned != manifest.total_tokens:
            warnings.append(
                f"rounded targets sum to {assigned}, manifest says {manifest.total_tokens}"
            )
    return ManifestReport(
        name=manifest.name,
        total_tokens=manifest.total_tokens,
        seq_len=manifest.seq_len,
        entries=entries,
        errors=errors,
        warnings=warnings,
    )


def mix_plan(manifest: DataManifest) -> list[dict[str, Any]]:
    report = validate_manifest(manifest)
    if not report.ok:
        raise ValueError("invalid manifest:\n" + "\n".join(report.errors))
    return [
        {
            "name": entry.name,
            "domain": entry.domain,
            "license": entry.license,
            "target_tokens": entry.target_tokens,
            "epochs": entry.epochs,
        }
        for entry in report.entries
    ]
