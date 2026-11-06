"""Visualisation helpers for training runs (SVG charts + a static HTML report)."""

from __future__ import annotations

from .report import build_report, load_metrics
from .svg import PALETTE, Annotation, Chart, Series, format_tick, log_ticks, nice_ticks

__all__ = [
    "Annotation",
    "Chart",
    "PALETTE",
    "Series",
    "build_report",
    "format_tick",
    "load_metrics",
    "log_ticks",
    "nice_ticks",
]
