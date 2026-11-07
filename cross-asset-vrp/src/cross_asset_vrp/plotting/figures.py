"""One function per published figure. One series per panel: stacked panels, bars, boxes and
small multiples, never several lines sharing an axis. Bars carry direct value labels."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from cross_asset_vrp.config import Config

Palette = dict[str, str]
ESTIMATOR_NAMES = {
    "close_to_close": "Squared\nclose-to-close",
    "parkinson": "Parkinson",
    "garman_klass": "Garman-\nKlass",
    "rogers_satchell": "Rogers-\nSatchell",
    "yang_zhang": "Yang-Zhang",
    "open_to_close_squared": "Squared\nopen-to-close",
}
SIZING_NAMES = {
    "unconditional": "Unconditional",
    "premium": "Sized by the forecast premium",
    "slope": "Sized by the VIX9D / VIX3M slope",
}


def _style(p: Palette) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": p["surface"],
            "axes.facecolor": p["surface"],
            "savefig.facecolor": p["surface"],
            "axes.edgecolor": p["grid"],
            "axes.labelcolor": p["muted"],
            "axes.titlecolor": p["text"],
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": p["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": p["muted"],
            "ytick.color": p["muted"],
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": p["text"],
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
        }
    )


def _save(fig: Figure, *, out: Path, name: str, dpi: int) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return {"png": f"reports/figures/{name}.png", "svg": f"reports/figures/{name}.svg"}


def _bars(
    ax: Axes,
    *,
    labels: list[str],
    values: list[float],
    color: str | list[str],
    p: Palette,
    fmt: Callable[[float], str],
) -> None:
    x = np.arange(len(values))
    ax.bar(x, values, width=0.62, color=color, edgecolor=p["surface"], linewidth=2, zorder=2)
    high = max([0.0, *values])
    low = min([0.0, *values])
    pad = 0.14 * max(high - low, 1e-12)
    for xi, v in zip(x, values, strict=True):
        y, va = (v + pad / 4, "bottom") if v >= 0 else (v - pad / 4, "top")
        ax.text(float(xi), y, fmt(v), ha="center", va=va, fontsize=8.2, color=p["text"])
    ax.set_xticks(x, labels)
    ax.axhline(0.0, color=p["muted"], linewidth=0.8, zorder=1)
    if high == low:
        high = 1.0
    ax.set_ylim(low - pad if low < 0 else 0.0, high + pad if high > 0 else 0.0)


def _share_ylim(axes: Iterable[Axes]) -> None:
    """Put panels drawn one at a time on the union of their ranges (they must not be created
    with sharey, which would make every get_ylim return the last range set)."""
    axes = list(axes)
    lows, highs = zip(*(ax.get_ylim() for ax in axes), strict=True)
    low, high = min(lows), max(highs)
    margin = 0.06 * (high - low)
    low, high = (low - margin if low < 0 else low), (high + margin if high > 0 else high)
    for i, ax in enumerate(axes):
        ax.set_ylim(low, high)
        if i:
            ax.tick_params(labelleft=False)


def _labels(config: Config) -> dict[str, str]:
    return {a.key: a.label for a in config.assets}


def premium(tables: Path, config: Config, p: Palette) -> Figure | None:
    """Headline: the distribution of the ex-ante premium in volatility points, asset by asset."""
    q = pd.read_csv(tables / "premium_distribution.csv")
    q = q.loc[q["measure"] == "ex_ante_points"]
    if q.empty:
        return None
    names = _labels(config)
    order = [a.key for a in config.assets if a.key in set(q["asset"])]
    q = q.set_index("asset").loc[order]
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(order) + 1.4))
    y = np.arange(len(order))[::-1]
    for yi, (_, row) in zip(y, q.iterrows(), strict=True):
        ax.plot([row["p5"], row["p95"]], [yi, yi], color=p["muted"], linewidth=1.2, zorder=2)
        ax.barh(
            yi,
            row["p75"] - row["p25"],
            left=row["p25"],
            height=0.5,
            color=p["this_study"],
            zorder=3,
        )
        ax.plot(
            [row["p50"], row["p50"]],
            [yi - 0.25, yi + 0.25],
            color=p["surface"],
            linewidth=2.2,
            zorder=4,
        )
        ax.text(
            row["p95"],
            yi,
            f"  median {row['p50']:+.1f}",
            va="center",
            fontsize=8.2,
            color=p["text"],
        )
    ax.axvline(0.0, color=p["text"], linewidth=0.9, zorder=1)
    ax.set_yticks(y, [names[k] for k in order])
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Implied volatility minus the HAR forecast of realized volatility, points")
    ax.set_title(
        "Ex-ante variance risk premium by market: box 25th to 75th percentile, line 5th to 95th",
        loc="left",
    )
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + 0.18 * (right - left))
    fig.tight_layout()
    return fig


def calibration(tables: Path, config: Config, p: Palette) -> Figure:
    c = pd.read_csv(tables / "calibration.csv")
    window = config.realized.yang_zhang_window
    day = c.loc[c["level"] == "day"]
    win = c.loc[c["level"] == "window"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    _bars(
        axes[0],
        labels=[ESTIMATOR_NAMES[e] for e in day["estimator"]],
        values=list(day["bias"] * 100),
        color=p["benchmark"],
        p=p,
        fmt=lambda v: f"{v:+.1f}%",
    )
    axes[0].set_title("Daily bias against 5-minute realized variance", loc="left")
    axes[0].set_ylabel("Mean estimate over mean RV, minus 1, %")
    _bars(
        axes[1],
        labels=[ESTIMATOR_NAMES[e] for e in win["estimator"]],
        values=list(win["efficiency_vs_cc"]),
        color=p["this_study"],
        p=p,
        fmt=lambda v: f"{v:.1f}x",
    )
    axes[1].set_title(f"Efficiency over {window}-day windows, against squared returns", loc="left")
    axes[1].set_ylabel("MSE of squared returns over the estimator's MSE")
    fig.tight_layout()
    return fig


def heston_check(tables: Path, p: Palette) -> Figure:
    h = pd.read_csv(tables / "heston.csv")
    steps = sorted(h["step_minutes"].unique())
    fig, axes = plt.subplots(1, len(steps), figsize=(4.4 * len(steps), 3.6), squeeze=False)
    for ax, step in zip(axes[0], steps, strict=True):
        g = h.loc[h["step_minutes"] == step]
        _bars(
            ax,
            labels=[ESTIMATOR_NAMES[e] for e in g["estimator"]],
            values=list(g["bias"] * 100),
            color=p["benchmark"],
            p=p,
            fmt=lambda v: f"{v:+.1f}%",
        )
        ax.set_title(f"Price sampled every {step} minute{'s' if step > 1 else ''}", loc="left")
    axes[0][0].set_ylabel("Bias against true integrated variance, %")
    _share_ylim(axes[0])
    fig.tight_layout()
    return fig


def futures_equity(tables: Path, config: Config, p: Palette) -> Figure:
    e = pd.read_csv(tables / "futures_equity.csv", parse_dates=["date"]).set_index("date")
    fig, axes = plt.subplots(len(e.columns), 1, figsize=(10.5, 2.5 * len(e.columns)), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), e.columns, strict=True):
        # An account that reached zero holds cash from then on; a log axis cannot show zero,
        # so the line stops on the last solvent day and the wipe-out is marked.
        # Days missing from this rule's series are NaN in the shared table; only a value at or
        # below zero marks the wipe-out.
        series = e[name].dropna()
        solvent = series.where(series > 0)
        ax.plot(solvent.index, solvent, color=p["this_study"], linewidth=1.3)
        ax.set_yscale("log")
        if (series <= 0).any():
            ruin = pd.Timestamp(str(series.index[int(np.flatnonzero(series <= 0)[0])]))
            ax.axvline(ruin, color=p["benchmark"], linewidth=1.2)
            ax.annotate(
                f"account lost {ruin.date()}",
                xy=(ruin, 0.5),
                xycoords=("data", "axes fraction"),
                xytext=(6, 0),
                textcoords="offset points",
                color=p["benchmark"],
                fontsize=9,
                va="center",
            )
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        for ep in config.episodes:
            start, end = pd.Timestamp(ep.start), pd.Timestamp(ep.end)
            if end < e.index[0] or start > e.index[-1]:
                continue
            ax.axvspan(
                max(start, e.index[0]),
                min(end, e.index[-1]),
                color=p["premium"],
                alpha=0.25,
                linewidth=0,
            )
        ax.set_title(f"{SIZING_NAMES[name]}: growth of 1", loc="left")
    shown = [
        ep.name
        for ep in config.episodes
        if pd.Timestamp(ep.end) >= e.index[0] and pd.Timestamp(ep.start) <= e.index[-1]
    ]
    fig.text(
        0.01,
        -0.01,
        "Shaded: " + ", ".join(shown) + ". Log scale; excess of T-bills, after roll costs.",
        fontsize=8,
        color=p["muted"],
    )
    fig.tight_layout()
    return fig


def predictability(tables: Path, config: Config, p: Palette) -> Figure | None:
    r = pd.read_csv(tables / "predictability.csv")
    r = r.loc[r["sample"] == "full"]
    if r.empty:
        return None
    names = _labels(config)
    horizons = sorted(r["horizon"].unique())
    order = [a.key for a in config.assets if a.key in set(r["asset"])]
    fig, axes = plt.subplots(
        1, len(horizons), figsize=(5.4 * len(horizons), 0.55 * len(order) + 1.6), squeeze=False
    )
    for ax, h in zip(axes[0], horizons, strict=True):
        g = r.loc[r["horizon"] == h].set_index("asset").reindex(order)
        y = np.arange(len(order))[::-1]
        for yi, (_, row) in zip(y, g.iterrows(), strict=True):
            if not np.isfinite(row["slope"]):
                continue
            ax.plot(
                [row["ci_low"] * 100, row["ci_high"] * 100],
                [yi, yi],
                color=p["muted"],
                linewidth=1.3,
            )
            ax.scatter([row["slope"] * 100], [yi], color=p["this_study"], s=34, zorder=3)
            ax.text(
                row["ci_high"] * 100,
                yi,
                f"  {row['slope'] * 100:+.1f}",
                va="center",
                fontsize=8,
                color=p["text"],
            )
        ax.axvline(0.0, color=p["text"], linewidth=0.9)
        ax.set_yticks(y, [names[k] for k in order])
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Next {h} month{'s' if h > 1 else ''}", loc="left")
        ax.set_xlabel("Annualized excess return per 1 s.d. of premium, % (95% EWC interval)")
        left, right = ax.get_xlim()
        ax.set_xlim(left, right + 0.15 * (right - left))
    for ax in axes[0][1:]:
        ax.tick_params(labelleft=False)
    fig.tight_layout()
    return fig


def replication_figure(tables: Path, p: Palette) -> Figure | None:
    r = pd.read_csv(tables / "replication.csv", parse_dates=["snapshot"])
    gap = r.dropna(subset=["gap_points"])
    pnl = r.dropna(subset=["short_pnl_vega"])
    if gap.empty and pnl.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.4), sharex=True)
    width = 20.0
    if len(gap):
        axes[0].bar(gap["snapshot"], gap["gap_points"], width=width, color=p["this_study"])
    axes[0].axhline(0.0, color=p["muted"], linewidth=0.8)
    axes[0].set_title(
        "Replicated 30-day volatility (mid) minus DVOL at the same minute, points", loc="left"
    )
    if len(pnl):
        colors = np.where(pnl["short_pnl_vega"] >= 0, p["this_study"], p["benchmark"])
        axes[1].bar(pnl["snapshot"], pnl["short_pnl_vega"], width=width, color=colors)
    axes[1].axhline(0.0, color=p["muted"], linewidth=0.8)
    axes[1].set_title(
        "Short 30-day variance swap sold at the bid: profit per unit of vega, volatility points",
        loc="left",
    )
    fig.tight_layout()
    return fig


def har_skill(results: dict[str, Any], config: Config, p: Palette) -> Figure | None:
    prem = results["estimates"]["premium"]
    days = config.windows.variance_days
    keys = [a.key for a in config.assets if prem[a.key].get("har_r2_vs_persistence") is not None]
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.6))
    _bars(
        ax,
        labels=[_labels(config)[k].replace(" ", "\n", 1) for k in keys],
        values=[100 * prem[k]["har_r2_vs_persistence"] for k in keys],
        color=p["this_study"],
        p=p,
        fmt=lambda v: f"{v:+.0f}%",
    )
    ax.set_title(
        f"HAR forecast of the next {days} days' variance: out-of-sample MSE cut against the past "
        f"{days} days",
        loc="left",
    )
    ax.set_ylabel("%")
    fig.tight_layout()
    return fig


def draw_all(
    *, config: Config, results: dict[str, Any], tables: Path, out: Path
) -> dict[str, dict[str, str]]:
    p = config.plot.palette
    _style(p)
    drawn: dict[str, Figure | None] = {
        "premium": premium(tables, config, p),
        "calibration": calibration(tables, config, p),
        "heston": heston_check(tables, p),
        "futures": futures_equity(tables, config, p),
        "predictability": predictability(tables, config, p),
        "replication": replication_figure(tables, p),
        "har": har_skill(results, config, p),
    }
    return {
        name: _save(fig, out=out, name=name, dpi=config.plot.dpi)
        for name, fig in drawn.items()
        if fig is not None
    }
