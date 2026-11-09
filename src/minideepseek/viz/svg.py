"""Dependency-free SVG line charts.

The training dashboard only needs a handful of line/scatter panels with a few
thousand points, so the project ships a tiny renderer instead of pulling in
matplotlib.  The output is a plain ``<svg>`` string that can be embedded in an
HTML report or written to a standalone file.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

#: Colour-blind-friendly categorical palette (Tableau 10-ish).
PALETTE = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#475569",
    "#b45309",
)


@dataclass
class Series:
    """One named line/scatter series in a chart."""

    label: str
    xs: Sequence[float]
    ys: Sequence[float]
    color: str = PALETTE[0]
    width: float = 2.0
    dash: str | None = None
    kind: str = "line"  # line | points


@dataclass
class Annotation:
    """Vertical reference line, e.g. the Chinchilla token budget."""

    x: float
    label: str
    color: str = "#94a3b8"


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Return ``count``-ish human-friendly tick values inside ``[low, high]``."""
    if not (_finite(low) and _finite(high)) or high <= low:
        return [low] if _finite(low) else []
    span = high - low
    raw_step = span / max(count, 1)
    exponent = math.floor(math.log10(raw_step))
    base = 10.0**exponent
    step = base
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiplier * base
        if span / step <= count:
            break
    start = math.ceil(low / step) * step
    ticks: list[float] = []
    value = start
    # Guard against floating-point drift at the end of the range.
    while value <= high + step * 1e-9 and len(ticks) < 100:
        ticks.append(round(value, 12))
        value += step
    return ticks


def log_ticks(low: float, high: float) -> list[float]:
    """Decade ticks (1, 10, 100, ...) covering ``[low, high]``."""
    if low <= 0 or high <= 0 or high <= low:
        return nice_ticks(low, high)
    start = math.floor(math.log10(low))
    stop = math.ceil(math.log10(high))
    ticks = [10.0**power for power in range(start, stop + 1)]
    return [tick for tick in ticks if low * 0.999 <= tick <= high * 1.001]


def format_tick(value: float) -> str:
    """Compact tick label: 0.001, 0.12, 3.4, 120, 1.2e+04."""
    if not _finite(value):
        return ""
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude >= 1e4 or magnitude < 1e-3:
        return f"{value:.1e}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def _downsample(series: Series, limit: int = 1500) -> Series:
    """Stride long series so the SVG stays small; keep the last point."""
    count = len(series.xs)
    if count <= limit:
        return series
    stride = math.ceil(count / limit)
    indices = list(range(0, count, stride))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return Series(
        label=series.label,
        xs=[series.xs[index] for index in indices],
        ys=[series.ys[index] for index in indices],
        color=series.color,
        width=series.width,
        dash=series.dash,
        kind=series.kind,
    )


@dataclass
class Chart:
    """A single SVG line chart with optional annotations."""

    title: str
    xlabel: str = "step"
    ylabel: str = ""
    width: int = 640
    height: int = 280
    logx: bool = False
    logy: bool = False
    series: list[Series] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)

    def add(self, series: Series) -> Chart:
        self.series.append(series)
        return self

    def add_series(
        self,
        label: str,
        xs: Sequence[float],
        ys: Sequence[float],
        *,
        color: str | None = None,
        kind: str = "line",
        dash: str | None = None,
        width: float = 2.0,
    ) -> Chart:
        color = color or PALETTE[len(self.series) % len(PALETTE)]
        return self.add(
            Series(
                label=label,
                xs=list(xs),
                ys=list(ys),
                color=color,
                kind=kind,
                dash=dash,
                width=width,
            )
        )

    # -- internals --------------------------------------------------------
    def _transform(self, value: float, log: bool) -> float:
        return math.log10(value) if log else float(value)

    def _domain(
        self, values: Sequence[float], log: bool
    ) -> tuple[float, float] | None:
        usable = [
            value
            for value in values
            if _finite(value) and (not log or value > 0)
        ]
        if not usable:
            return None
        low, high = min(usable), max(usable)
        if log:
            low, high = math.log10(low), math.log10(high)
        if high - low < 1e-12:
            if not log and low == 0:
                # A flat-zero series (e.g. clamp rate before clamping starts)
                # reads best on a 0–1 axis rather than ±1.
                return 0.0, 1.0
            pad = max(abs(high) * 0.05, 0.05 if log else 0.1)
            return low - pad, high + pad
        pad = (high - low) * 0.06
        return low - pad, high + pad

    def render(self) -> str:
        width, height = self.width, self.height
        left, right, top, bottom = 68, 16, 42, 42
        plot_width = width - left - right
        plot_height = height - top - bottom

        prepared = [_downsample(series) for series in self.series]
        x_values = [x for series in prepared for x in series.xs]
        y_values = [y for series in prepared for y in series.ys]
        x_domain = self._domain(x_values, self.logx)
        y_domain = self._domain(y_values, self.logy)
        if x_domain is None or y_domain is None:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
                f'role="img" aria-label="{html.escape(self.title)}">'
                f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
                f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
                f'fill="#94a3b8" font-size="13">{html.escape(self.title)} — no data</text></svg>'
            )

        x_low, x_high = x_domain
        y_low, y_high = y_domain

        def px(value: float) -> float:
            position = (self._transform(value, self.logx) - x_low) / (x_high - x_low)
            return left + position * plot_width

        def py(value: float) -> float:
            position = (self._transform(value, self.logy) - y_low) / (y_high - y_low)
            return top + (1.0 - position) * plot_height

        parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="100%" role="img" aria-label="{html.escape(self.title)}" '
            f'font-family="ui-sans-serif, -apple-system, Segoe UI, sans-serif">',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            f'<text x="{left}" y="22" font-size="14" font-weight="600" fill="#0f172a">'
            f"{html.escape(self.title)}</text>",
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
            f'fill="#fbfdff" stroke="#e2e8f0"/>',
        ]

        # Horizontal grid + y ticks.
        y_ticks = (
            log_ticks(10**y_low, 10**y_high)
            if self.logy
            else nice_ticks(y_low, y_high)
        )
        for tick in y_ticks:
            position = py(tick)
            if not (top - 0.5 <= position <= top + plot_height + 0.5):
                continue
            parts.append(
                f'<line x1="{left}" y1="{position:.2f}" x2="{left + plot_width}" '
                f'y2="{position:.2f}" stroke="#eef2f7"/>'
            )
            parts.append(
                f'<text x="{left - 8}" y="{position + 3.5:.2f}" text-anchor="end" '
                f'font-size="10" fill="#64748b">{format_tick(tick)}</text>'
            )

        # Vertical grid + x ticks.
        x_ticks = (
            log_ticks(10**x_low, 10**x_high)
            if self.logx
            else nice_ticks(x_low, x_high, count=6)
        )
        for tick in x_ticks:
            position = px(tick)
            if not (left - 0.5 <= position <= left + plot_width + 0.5):
                continue
            parts.append(
                f'<line x1="{position:.2f}" y1="{top}" x2="{position:.2f}" '
                f'y2="{top + plot_height}" stroke="#f1f5f9"/>'
            )
            parts.append(
                f'<text x="{position:.2f}" y="{top + plot_height + 16}" '
                f'text-anchor="middle" font-size="10" fill="#64748b">{format_tick(tick)}</text>'
            )

        # Annotations (e.g. token-budget markers).
        for annotation in self.annotations:
            if not _finite(annotation.x) or annotation.x <= 0:
                continue
            position = px(annotation.x)
            if not (left <= position <= left + plot_width):
                continue
            parts.append(
                f'<line x1="{position:.2f}" y1="{top}" x2="{position:.2f}" '
                f'y2="{top + plot_height}" stroke="{annotation.color}" '
                f'stroke-dasharray="5 4" stroke-width="1.2"/>'
            )
            near_right = position > left + plot_width - 90
            text_x = position - 4 if near_right else position + 4
            anchor = "end" if near_right else "start"
            parts.append(
                f'<text x="{text_x:.2f}" y="{top + 12}" text-anchor="{anchor}" '
                f'font-size="10" fill="{annotation.color}">'
                f"{html.escape(annotation.label)}</text>"
            )

        # Data.
        for index, series in enumerate(prepared):
            color = series.color or PALETTE[index % len(PALETTE)]
            if series.kind == "points":
                for x, y in zip(series.xs, series.ys, strict=False):
                    if not (_finite(x) and _finite(y)) or (self.logy and y <= 0):
                        continue
                    parts.append(
                        f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="2.6" '
                        f'fill="{color}" fill-opacity="0.85"/>'
                    )
                continue
            run: list[str] = []
            for x, y in zip(series.xs, series.ys, strict=False):
                if not (_finite(x) and _finite(y)) or (self.logy and y <= 0):
                    if len(run) > 1:
                        parts.append(self._polyline(run, color, series))
                    run = []
                    continue
                run.append(f"{px(x):.2f},{py(y):.2f}")
            if len(run) > 1:
                parts.append(self._polyline(run, color, series))

        # Legend.
        legend_x = left
        for index, series in enumerate(prepared):
            color = series.color or PALETTE[index % len(PALETTE)]
            label = series.label if len(series.label) <= 22 else series.label[:21] + "…"
            parts.append(
                f'<line x1="{legend_x}" y1="34" x2="{legend_x + 16}" y2="34" '
                f'stroke="{color}" stroke-width="2.4"/>'
            )
            parts.append(
                f'<text x="{legend_x + 20}" y="37.5" font-size="10.5" fill="#334155">'
                f"{html.escape(label)}</text>"
            )
            legend_x += 30 + 6.2 * len(label)
            if legend_x > width - 90:
                break

        # Axis labels.
        parts.append(
            f'<text x="{left + plot_width / 2}" y="{height - 8}" text-anchor="middle" '
            f'font-size="11" fill="#475569">{html.escape(self.xlabel)}</text>'
        )
        if self.ylabel:
            parts.append(
                f'<text x="16" y="{top + plot_height / 2}" text-anchor="middle" '
                f'font-size="11" fill="#475569" transform="rotate(-90 16 '
                f'{top + plot_height / 2:.1f})">{html.escape(self.ylabel)}</text>'
            )
        parts.append("</svg>")
        return "".join(parts)

    @staticmethod
    def _polyline(points: list[str], color: str, series: Series) -> str:
        dash = f' stroke-dasharray="{series.dash}"' if series.dash else ""
        return (
            f'<polyline fill="none" stroke="{color}" stroke-width="{series.width}" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash} '
            f'points="{" ".join(points)}"/>'
        )


def _mix(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> str:
    """Linear blend between two RGB colours, ``t`` in [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    red, green, blue = (round(low[index] + (high[index] - low[index]) * t) for index in range(3))
    return f"#{red:02x}{green:02x}{blue:02x}"


_WHITE = (255, 255, 255)
_BLUE = (37, 99, 235)
_RED = (220, 38, 38)


@dataclass
class Heatmap:
    """A row x column grid of coloured cells (expert load, enrichment, ...).

    ``center`` turns the colour scale diverging around that value, which is what
    makes a load ratio of 1.0 (perfect balance) or an enrichment of 1.0 (no
    preference) readable at a glance.
    """

    title: str
    row_labels: Sequence[str]
    col_labels: Sequence[str]
    values: Sequence[Sequence[float]]
    xlabel: str = ""
    ylabel: str = ""
    width: int = 640
    height: int = 280
    center: float | None = None
    #: Smallest half-width of a diverging scale.  Without it a metric that only
    #: wobbles (enrichment 0.97-1.03) saturates the colour map and reads as a
    #: dramatic effect; with it, colour distance means a fixed effect size.
    min_span: float | None = None
    fmt: str = "{:.2f}"

    def _bounds(self) -> tuple[float, float]:
        flat = [v for row in self.values for v in row if _finite(v)]
        if not flat:
            return 0.0, 1.0
        low, high = min(flat), max(flat)
        if self.center is not None:
            span = max(abs(high - self.center), abs(self.center - low)) or 1e-9
            if self.min_span is not None:
                span = max(span, self.min_span)
            return self.center - span, self.center + span
        return low, high if high > low else low + 1.0

    def _colour(self, value: float, low: float, high: float) -> str:
        if not _finite(value):
            return "#f1f5f9"
        if self.center is None:
            return _mix(_WHITE, _BLUE, (value - low) / (high - low))
        span = (high - low) / 2 or 1.0
        offset = (value - self.center) / span
        if offset < 0:
            return _mix(_WHITE, _BLUE, -offset)
        return _mix(_WHITE, _RED, offset)

    def render(self) -> str:
        width, height = self.width, self.height
        rows, cols = len(self.row_labels), len(self.col_labels)
        if rows == 0 or cols == 0 or not self.values:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
                f'role="img" aria-label="{html.escape(self.title)}">'
                f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
                f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
                f'fill="#94a3b8" font-size="13">{html.escape(self.title)} — no data</text></svg>'
            )
        label_px = max((len(label) for label in self.row_labels), default=4)
        left = min(max(16 + 6.4 * label_px, 64), 190)
        top, right, bottom = 30, 16, 44
        plot_width = width - left - right
        plot_height = height - top - bottom
        cell_w = plot_width / cols
        cell_h = plot_height / rows
        low, high = self._bounds()

        parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="100%" role="img" aria-label="{html.escape(self.title)}" '
            f'font-family="ui-sans-serif, -apple-system, Segoe UI, sans-serif">',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            f'<text x="{left}" y="18" font-size="14" font-weight="600" fill="#0f172a">'
            f"{html.escape(self.title)}</text>",
        ]
        # Cells.
        for row_index, row in enumerate(self.values):
            for col_index in range(cols):
                value = row[col_index] if col_index < len(row) else float("nan")
                x = left + col_index * cell_w
                y = top + row_index * cell_h
                parts.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + 0.5:.2f}" '
                    f'height="{cell_h + 0.5:.2f}" fill="{self._colour(value, low, high)}" '
                    f'stroke="#ffffff" stroke-width="1"/>'
                )
                if _finite(value) and cell_w >= 30 and cell_h >= 13:
                    text_fill = "#0f172a"
                    magnitude = abs(value - (self.center or 0.0)) / max(
                        abs(high - low) / 2, 1e-9
                    )
                    if magnitude > 0.55:
                        text_fill = "#ffffff"
                    parts.append(
                        f'<text x="{x + cell_w / 2:.2f}" y="{y + cell_h / 2 + 3.5:.2f}" '
                        f'text-anchor="middle" font-size="9.5" fill="{text_fill}">'
                        f"{html.escape(self.fmt.format(value))}</text>"
                    )
        # Row labels.
        for row_index, label in enumerate(self.row_labels):
            parts.append(
                f'<text x="{left - 6}" y="{top + row_index * cell_h + cell_h / 2 + 3.5:.2f}" '
                f'text-anchor="end" font-size="10" fill="#475569">'
                f"{html.escape(label)}</text>"
            )
        # Column labels, skipped when they would overlap.
        step = max(1, math.ceil(22 / max(cell_w, 1)))
        for col_index, label in enumerate(self.col_labels):
            if col_index % step:
                continue
            parts.append(
                f'<text x="{left + col_index * cell_w + cell_w / 2:.2f}" '
                f'y="{top + plot_height + 13}" text-anchor="middle" font-size="10" '
                f'fill="#475569">{html.escape(label)}</text>'
            )
        # Axis titles + colour scale legend.
        if self.xlabel:
            parts.append(
                f'<text x="{left + plot_width / 2}" y="{height - 8}" text-anchor="middle" '
                f'font-size="11" fill="#475569">{html.escape(self.xlabel)}</text>'
            )
        if self.ylabel:
            parts.append(
                f'<text x="14" y="{top + plot_height / 2}" text-anchor="middle" '
                f'font-size="11" fill="#475569" transform="rotate(-90 14 '
                f'{top + plot_height / 2:.1f})">{html.escape(self.ylabel)}</text>'
            )
        # Colour scale range, bottom-right so it never collides with xlabel.
        parts.append(
            f'<text x="{left + plot_width}" y="{height - 8}" text-anchor="end" '
            f'font-size="9.5" fill="#94a3b8">{html.escape(self.fmt.format(low))} … '
            f"{html.escape(self.fmt.format(high))}</text>"
        )
        parts.append("</svg>")
        return "".join(parts)
