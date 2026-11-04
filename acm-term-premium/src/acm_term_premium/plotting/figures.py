"""Published figures. One function per figure; all share the style set here.

Series colors come from a colorblind-validated categorical palette, assigned
to fixed roles (this study, the NY Fed, expectations, term premium) so a color
means the same thing in every figure. Every figure has a table twin in
reports/tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # backend is chosen before pyplot loads
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
THIS_STUDY = "#2a78d6"
NY_FED = "#eb6834"
EXPECTATIONS = "#1baf7a"
PREMIUM = "#eda100"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
        }
    )


def _save(fig: Figure, directory: Path, stem: str, dpi: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{stem}.png", directory / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(paths[1], bbox_inches="tight", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    return paths


def replication(
    mine: pd.Series, published: pd.Series, years: int, *, out: Path, dpi: int
) -> list[Path]:
    """This study's term premium (top) and its difference from the published series (bottom).

    Two series that agree to a hundredth of a basis point cannot be told apart
    when drawn together, so each gets its own panel.
    """
    _style()
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(9.6, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.12},
    )
    top.plot(mine.index, mine * 100, color=THIS_STUDY)
    top.axhline(0, color=MUTED, linewidth=0.7)
    top.set_ylabel("Term premium (%)")
    top.set_title(f"{years}-year Treasury term premium, rebuilt from public inputs")
    residual = (mine - published.reindex(mine.index)).abs() * 1e4
    # Bars of the worst month in each year: a line of near-zero differences
    # would sit on the axis, while bars stay legible at any magnitude.
    yearly = residual.groupby(pd.DatetimeIndex(residual.index).year).max()
    bottom.bar(
        pd.to_datetime([f"{year}-07-01" for year in yearly.index]),
        yearly.to_numpy(),
        width=300,
        color=NY_FED,
    )
    bottom.set_ylabel("Largest |gap| to\nNY Fed (bp)")
    bottom.set_ylim(0, max(float(yearly.max()) * 1.35, 0.01))
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    bottom.text(
        0.01,
        0.82,
        f"Largest gap in any month: {float(residual.max()):.4f} bp",
        transform=bottom.transAxes,
        fontsize=8,
        color=MUTED,
    )
    return _save(fig, out, "replication", dpi)


def decomposition(
    fitted: pd.Series,
    neutral: pd.Series,
    premium: pd.Series,
    *,
    years: int,
    lower: pd.Series,
    upper: pd.Series,
    out: Path,
    dpi: int,
) -> list[Path]:
    """Yield, expected short rates and term premium in three stacked panels."""
    _style()
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 7.4), sharex=True, gridspec_kw={"hspace": 0.18})
    panels = (
        (fitted, TEXT, f"{years}-year yield (%)"),
        (neutral, EXPECTATIONS, "Expected average\nshort rate (%)"),
        (premium, PREMIUM, "Term premium (%)"),
    )
    for ax, (series, color, label) in zip(axes, panels, strict=True):
        ax.plot(series.index, series * 100, color=color)
        ax.set_ylabel(label)
    axes[2].fill_between(
        premium.index, lower * 100, upper * 100, color=PREMIUM, alpha=0.22, linewidth=0
    )
    axes[2].axhline(0, color=MUTED, linewidth=0.7)
    axes[2].text(
        0.01,
        0.06,
        "Shaded: 90% bootstrap band",
        transform=axes[2].transAxes,
        fontsize=8,
        color=MUTED,
    )
    axes[0].set_title(f"The {years}-year yield: expected short rates plus term premium")
    return _save(fig, out, "decomposition", dpi)


def _runs(index: pd.DatetimeIndex, flags: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous flagged months as (start, end) spans that cover each whole month."""
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: pd.Timestamp | None = None
    for i, flagged in enumerate(flags):
        if flagged and start is None:
            start = index[i] - pd.offsets.MonthEnd(1)
        if start is not None and (not flagged or i == len(flags) - 1):
            spans.append((start, index[i if flagged else i - 1]))
            start = None
    return spans


def revisions(
    realtime: pd.Series,
    unconstrained: pd.Series,
    full: pd.Series,
    constrained: pd.Series,
    *,
    years: int,
    out: Path,
    dpi: int,
) -> list[Path]:
    """Real-time minus full-sample term premium; the unconstrained path in its own panel."""
    _style()
    diff = (realtime - full.reindex(realtime.index)) * 1e4
    raw = (unconstrained - full.reindex(unconstrained.index)) * 1e4
    bound = constrained.reindex(diff.index).fillna(False).to_numpy(dtype=bool)
    rows = 2 if bound.any() else 1
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(9.6, 4.4 if rows == 1 else 6.6),
        sharex=True,
        squeeze=False,
        gridspec_kw={"hspace": 0.2},
    )
    panels = [(diff, THIS_STUDY, "Constrained VAR (bp)")]
    if bound.any():
        panels.append((raw, MUTED, "Unconstrained VAR (bp)"))
    spans = _runs(pd.DatetimeIndex(diff.index), bound)
    for ax, (series, color, label) in zip(axes[:, 0], panels, strict=True):
        for first, last in spans:
            ax.axvspan(first, last, color=GRID, linewidth=0)
        ax.plot(series.index, series, color=color)
        ax.axhline(0, color=MUTED, linewidth=0.7)
        ax.set_ylabel(label)
    axes[0, 0].set_title(f"Real-time minus full-sample {years}-year term premium")
    if spans:
        fig.text(
            0.01,
            -0.01,
            "Shaded months: largest VAR root above the bound; the top panel "
            "scales those roots to the bound.",
            fontsize=8,
            color=MUTED,
        )
    return _save(fig, out, "revisions", dpi)


def forecast_r2(stats: pd.DataFrame, *, out: Path, dpi: int) -> list[Path]:
    """Out-of-sample R^2 by bond maturity, one panel per predictor.

    Each bar is labelled with its R^2 and its one-sided Clark-West p-value, so a
    bar too short to see still carries its number.
    """
    _style()
    maturities = sorted(stats["maturity"].unique())
    titles = {"acm": "ACM term premium (real time)", "cp": "Cochrane-Piazzesi factor"}
    colors = {"acm": THIS_STUDY, "cp": MUTED}
    values = stats["oos_r2"].to_numpy(dtype=float) * 100
    low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
    pad = 0.28 * (high - low)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    x = np.arange(len(maturities))
    for ax, predictor in zip(axes, ["acm", "cp"], strict=True):
        rows = stats[stats["predictor"] == predictor].set_index("maturity").reindex(maturities)
        r2 = rows["oos_r2"].to_numpy(dtype=float) * 100
        bars = ax.bar(x, r2, 0.6, color=colors[predictor])
        for bar, value, p in zip(bars, r2, rows["p_value"].to_numpy(dtype=float), strict=True):
            above = value >= 0
            ax.annotate(
                f"{value:+.2f}%\np = {p:.2f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4 if above else -4),
                textcoords="offset points",
                ha="center",
                va="bottom" if above else "top",
                fontsize=8.5,
                color=TEXT,
            )
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xticks(x, [f"{m // 12}-year" for m in maturities])
        ax.set_title(titles[predictor], fontsize=11)
    axes[0].set_ylabel("Out-of-sample R² vs historical mean (%)")
    fig.suptitle(
        "Forecasting 12-month bond excess returns out of sample",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.01,
        -0.03,
        "p: one-sided Clark-West p-value against the historical mean (Newey-West errors).",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout()
    return _save(fig, out, "forecasts", dpi)


def policy_days(summary: dict[str, Any], *, years: int, out: Path, dpi: int) -> list[Path]:
    """Policy-rate change days against every trading day of the same period.

    Left: mean absolute daily change in the yield and its two components. Right:
    share of the variance of yield changes from each component. Bars, with the
    value printed, so no marker or line can hide another.
    """
    _style()
    every = summary["all_days"]
    groups = [
        (f"Policy-change days ({summary['events']})", THIS_STUDY, summary),
        (f"All days ({every['days']:,})", MUTED, every),
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    width = 0.38
    panels = (
        (
            left,
            ["mean_abs_yield_bp", "mean_abs_expectations_bp", "mean_abs_term_premium_bp"],
            [f"{years}-year yield", "Expected\nshort rates", "Term premium"],
            1.0,
            "{:.1f}",
            "Mean absolute daily change (bp)",
        ),
        (
            right,
            ["expectations_variance_share", "term_premium_variance_share"],
            ["Expected\nshort rates", "Term premium"],
            100.0,
            "{:.0f}%",
            "Share of yield-change variance (%)",
        ),
    )
    for ax, keys, labels, scale, fmt, title in panels:
        x = np.arange(len(keys))
        top = 0.0
        for i, (name, color, source) in enumerate(groups):
            vals = [scale * float(source[k]) for k in keys]
            top = max(top, *vals)
            bars = ax.bar(x + (i - 0.5) * width, vals, width - 0.04, color=color, label=name)
            for bar, value in zip(bars, vals, strict=True):
                ax.annotate(
                    fmt.format(value),
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=TEXT,
                )
        ax.set_ylim(0, top * 1.2)
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", visible=False)
    left.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncols=2)
    fig.suptitle(
        f"The {years}-year yield on policy-rate change days",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=TEXT,
    )
    fig.tight_layout()
    return _save(fig, out, "policy_days", dpi)
