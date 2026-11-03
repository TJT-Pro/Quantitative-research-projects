"""Published figures. Each function takes tidy tables and writes PNG + SVG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter

from covariance_horse_race.plotting import style
from covariance_horse_race.plotting.style import ModelStyle, plt


def _plain_log_axis(axis: object, ticks: list[float] | None = None) -> None:
    """Log axis with plain-number tick labels instead of scientific notation."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))  # type: ignore[attr-defined]
    axis.set_minor_formatter(NullFormatter())  # type: ignore[attr-defined]
    if ticks is None:
        # 1-2-5 ticks spanning the data range, so every decade is labelled.
        lo, hi = axis.get_data_interval()  # type: ignore[attr-defined]
        if lo <= 0 or not np.isfinite(lo) or not np.isfinite(hi):
            return
        decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
        ticks = [
            m * 10.0**k for k in decades for m in (1, 2, 5) if lo * 0.8 <= m * 10.0**k <= hi * 1.25
        ]
    axis.set_ticks(ticks)  # type: ignore[attr-defined]


def mcs_heatmap(
    p_values: pd.DataFrame,
    *,
    styles: dict[str, ModelStyle],
    alpha: float,
    title: str,
    out: Path,
    stem: str,
    dpi: int,
) -> list[Path]:
    """Rows are models, columns are periods, cells are MCS p-values.

    Cells inside the confidence set (p >= alpha) are filled on the sequential ramp
    by p-value and labelled with the value; cells outside are left on the surface
    with a muted dash, so membership reads without relying on color.
    """
    style.apply()
    models = list(p_values.index)
    periods = list(p_values.columns)
    fig_h = 0.42 * len(models) + 1.6
    fig, ax = plt.subplots(figsize=(max(6.0, 0.85 * len(periods) + 2.6), fig_h))
    cmap = ListedColormap(style.SEQUENTIAL)
    shown = p_values.where(p_values >= alpha)
    ax.imshow(shown.to_numpy(dtype=float), cmap=cmap, vmin=alpha, vmax=1.0, aspect="auto")
    for i, model in enumerate(models):
        for j, period in enumerate(periods):
            value = p_values.loc[model, period]
            if pd.isna(value):
                text, colour = "n/a", style.TEXT_SECONDARY
            elif value >= alpha:
                text = f"{value:.2f}"
                colour = "white" if value > 0.55 else style.TEXT_PRIMARY
            else:
                text, colour = "out", style.TEXT_SECONDARY
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=colour)
    ax.set_xticks(range(len(periods)), periods)
    ax.set_yticks(range(len(models)), [styles[m].label for m in models])
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(periods)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
    ax.grid(which="minor", color=style.SURFACE, linewidth=2)
    ax.set_title(title)
    fig.text(
        0.01,
        -0.02,
        f"Cell value: MCS p-value. Shaded = in the {int((1 - alpha) * 100)}% Model Confidence Set; "
        "'out' = eliminated; 'n/a' = model not defined in every month of the period.",
        fontsize=8,
        color=style.TEXT_SECONDARY,
    )
    return style.save(fig, out, stem, dpi)


def _grid_heatmap(
    values: pd.DataFrame,
    *,
    groups: list[str],
    best_marker: bool,
    note: str,
    title: str,
    out: Path,
    stem: str,
    dpi: int,
) -> list[Path]:
    """Models in rows, (group, setting) pairs in columns, one labelled cell each.

    `values` is indexed by model label with a two-level column index (group,
    setting). Color follows log(1 + value) on the sequential ramp so a single
    badly conditioned model cannot wash out the rest, and every cell carries its
    number, so the figure reads without the color at all. Groups are separated
    by a surface-colored gap and named above their columns.
    """
    style.apply()
    n_rows, n_cols = values.shape
    grid = values.to_numpy(dtype=float)
    shade = np.log1p(np.where(np.isfinite(grid), np.clip(grid, 0.0, None), np.nan))
    top = float(np.nanmax(shade)) if np.isfinite(shade).any() else 1.0
    fig, ax = plt.subplots(figsize=(0.62 * n_cols + 2.8, 0.44 * n_rows + 2.0))
    cmap = LinearSegmentedColormap.from_list("ramp", style.SEQUENTIAL).with_extremes(
        bad=style.SURFACE
    )
    ax.imshow(np.ma.masked_invalid(shade), cmap=cmap, vmin=0.0, vmax=top, aspect="auto")
    for i in range(n_rows):
        for j in range(n_cols):
            value = grid[i, j]
            if not np.isfinite(value):
                ax.text(
                    j, i, "n/a", ha="center", va="center", fontsize=7.5, color=style.TEXT_SECONDARY
                )
                continue
            text = _cell_text(value, best_marker=best_marker)
            colour = "white" if shade[i, j] > 0.55 * top else style.TEXT_PRIMARY
            weight = "bold" if text == "best" else "normal"
            ax.text(
                j, i, text, ha="center", va="center", fontsize=8, color=colour, fontweight=weight
            )
    settings = [str(c[1]) for c in values.columns]
    ax.set_xticks(range(n_cols), settings)
    ax.set_yticks(range(n_rows), list(values.index))
    ax.tick_params(axis="both", which="both", length=0)
    ax.xaxis.tick_top()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color=style.SURFACE, linewidth=1.5)
    column_groups = [str(c[0]) for c in values.columns]
    for group in groups:
        cols = [j for j, g in enumerate(column_groups) if g == group]
        if not cols:
            continue
        ax.text(
            (cols[0] + cols[-1]) / 2,
            -1.25,
            group,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=style.TEXT_PRIMARY,
        )
        if cols[0] > 0:
            ax.axvline(cols[0] - 0.5, color=style.SURFACE, linewidth=6)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", color=style.TEXT_PRIMARY)
    fig.text(0.01, -0.02, note, fontsize=8, color=style.TEXT_SECONDARY)
    fig.tight_layout()
    return style.save(fig, out, stem, dpi)


def _cell_text(value: float, *, best_marker: bool) -> str:
    """Two decimals under 1, one under 100, none above, so small gaps stay visible."""
    if best_marker and value == 0.0:
        return "best"
    if value < 1:
        return f"{value:.2f}"
    return f"{value:.1f}" if value < 100 else f"{value:.0f}"


def _ordered_labels(styles: dict[str, ModelStyle], present: object) -> list[str]:
    """Model labels in config order, keeping only models present in the data."""
    names = set(present)  # type: ignore[call-overload]
    return [s.label for name, s in styles.items() if name in names]


def dimension_sweep(
    table: pd.DataFrame,
    *,
    styles: dict[str, ModelStyle],
    universe_labels: dict[str, str],
    out: Path,
    stem: str,
    dpi: int,
) -> list[Path]:
    """Excess out-of-sample GMV volatility over the best model in each
    (universe, window) cell, as a labelled heatmap.

    `table` has columns universe, window, model, excess_vol_pct. Columns run
    across universes, smallest to largest, and within each universe from the
    shortest to the longest estimation window.
    """
    universes = list(dict.fromkeys(table["universe"]))
    windows = sorted(int(w) for w in table["window"].unique())
    frame = table.assign(label=table["model"].map({k: v.label for k, v in styles.items()}))
    wide = frame.pivot_table(
        index="label", columns=["universe", "window"], values="excess_vol_pct", aggfunc="first"
    )
    order = [(u, w) for u in universes for w in windows if (u, w) in wide.columns]
    wide = wide.reindex(index=_ordered_labels(styles, table["model"]), columns=order)
    wide.columns = pd.MultiIndex.from_tuples([(universe_labels[u], f"{w}d") for u, w in order])
    return _grid_heatmap(
        wide,
        groups=[universe_labels[u] for u in universes],
        best_marker=True,
        note=(
            "Cell value: annualized GMV volatility above the best model in that column, in "
            "percent. 'best' = lowest volatility. Column labels: estimation window in trading "
            "days. 'n/a' = model not run at that window."
        ),
        title="Out-of-sample GMV volatility above the best model, by universe and window",
        out=out,
        stem=stem,
        dpi=dpi,
    )


def eigen_spectrum(
    sample: np.ndarray, shrunk: np.ndarray, *, title: str, out: Path, stem: str, dpi: int
) -> list[Path]:
    """Three panels sharing the rank axis: sample eigenvalues and nonlinear-shrinkage
    eigenvalues on the same log scale, and their ratio on a linear scale."""
    style.apply()
    rank = np.arange(1, sample.size + 1)
    sample_sorted = np.sort(sample)[::-1]
    shrunk_sorted = np.sort(shrunk)[::-1]
    ratio = shrunk_sorted / sample_sorted
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), sharex=True)
    panels = [
        (
            sample_sorted,
            style.FAMILY_COLORS["unconditional"],
            "Sample eigenvalues",
            "Daily variance (log)",
        ),
        (
            shrunk_sorted,
            style.FAMILY_COLORS["shrinkage"],
            "Nonlinear shrinkage",
            "Daily variance (log)",
        ),
        (ratio, style.TEXT_PRIMARY, "Shrunk / sample", "Ratio"),
    ]
    low = min(float(sample_sorted.min()), float(shrunk_sorted.min()))
    high = max(float(sample_sorted.max()), float(shrunk_sorted.max()))
    for index, (ax, (series, colour, name, ylabel)) in enumerate(zip(axes, panels, strict=True)):
        ax.plot(rank, series, color=colour)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Eigenvalue rank")
        ax.set_ylabel(ylabel)
        if index < 2:
            ax.set_yscale("log")
            ax.set_ylim(low * 0.7, high * 1.4)
        else:
            # Above 1 the eigenvalue was raised, below 1 lowered.
            ax.axhline(1.0, color=style.TEXT_SECONDARY, linewidth=0.8)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", color=style.TEXT_PRIMARY)
    fig.tight_layout()
    return style.save(fig, out, stem, dpi)


def turnover_vs_vol(
    table: pd.DataFrame,
    *,
    styles: dict[str, ModelStyle],
    title: str,
    out: Path,
    stem: str,
    dpi: int,
) -> list[Path]:
    """Two bar panels sharing the model axis: realized GMV volatility and average
    monthly turnover. `table` is indexed by model with columns vol_gmv, turnover
    (fractions). Models are sorted by volatility, lowest at the top.

    Separate panels keep models with near-identical risk and turnover readable,
    which a single scatter cannot. Turnover spans an order of magnitude, so its
    axis is log.
    """
    style.apply()
    ordered = [m for m in styles if m in table.index]
    rows = table.loc[ordered].sort_values("vol_gmv", ascending=False)
    labels = [styles[str(m)].label for m in rows.index]
    colours = [styles[str(m)].color for m in rows.index]
    positions = np.arange(len(rows))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 0.42 * len(rows) + 1.8), sharey=True)
    vol = rows["vol_gmv"].to_numpy(dtype=float) * 100
    turnover = rows["turnover"].to_numpy(dtype=float) * 100
    left.barh(positions, vol, color=colours, height=0.65)
    left.set_xlim(0, vol.max() * 1.15)
    for y, v in zip(positions, vol, strict=True):
        left.text(v, y, f" {v:.2f}", va="center", fontsize=8.5, color=style.TEXT_PRIMARY)
    left.set_yticks(positions, labels)
    left.set_xlabel("Annualized GMV volatility (%)")
    left.set_title("Risk", fontsize=11)
    right.barh(positions, turnover, color=colours, height=0.65)
    right.set_xscale("log")
    lo, hi = float(turnover.min()) * 0.5, float(turnover.max()) * 2.2
    right.set_xlim(lo, hi)
    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    _plain_log_axis(
        right.xaxis, [m * 10.0**k for k in decades for m in (1, 2, 5) if lo <= m * 10.0**k <= hi]
    )
    for y, v in zip(positions, turnover, strict=True):
        right.text(v, y, f" {v:.0f}", va="center", fontsize=8.5, color=style.TEXT_PRIMARY)
    right.set_xlabel("Average monthly turnover (% of portfolio, log scale)")
    right.set_title("Trading", fontsize=11)
    for ax in (left, right):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", color=style.TEXT_PRIMARY)
    fig.tight_layout()
    return style.save(fig, out, stem, dpi)


def simulation_panels(
    sim: pd.DataFrame, *, styles: dict[str, ModelStyle], out: Path, stem: str, dpi: int
) -> list[Path]:
    """Excess true GMV variance over the oracle, by dimension N and sample size T,
    as a labelled heatmap."""
    dims = sorted(int(n) for n in sim["n"].unique())
    sizes = sorted(int(t) for t in sim["t"].unique())
    frame = sim.assign(label=sim["model"].map({k: v.label for k, v in styles.items()}))
    wide = frame.pivot_table(
        index="label", columns=["n", "t"], values="excess_gmv_variance_pct", aggfunc="first"
    )
    order = [(n, t) for n in dims for t in sizes if (n, t) in wide.columns]
    wide = wide.reindex(index=_ordered_labels(styles, sim["model"]), columns=order)
    wide.columns = pd.MultiIndex.from_tuples([(f"N = {n}", f"T={t}") for n, t in order])
    groups = [f"N = {n}" for n in dims]
    return _grid_heatmap(
        wide,
        groups=groups,
        best_marker=False,
        note=(
            "Cell value: true variance of the estimated GMV portfolio above the oracle GMV "
            "variance, in percent, averaged over replications. N = assets, T = sample size "
            "in trading days."
        ),
        title="Known-covariance simulation: cost of estimation error",
        out=out,
        stem=stem,
        dpi=dpi,
    )
