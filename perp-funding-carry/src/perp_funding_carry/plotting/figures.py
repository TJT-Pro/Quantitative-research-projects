"""One function per published figure. One series per panel: stacked panels, bars, a
heatmap and small multiples, never several lines sharing an axis. Bars carry direct
value labels because two palette colors fall below 3:1 contrast on the surface."""

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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from perp_funding_carry.config import Config

Palette = dict[str, str]
MODEL_TITLES = {
    "persistence": "Persistence",
    "har": "HAR",
    "har_plus": "HAR + basis + momentum",
    "har_plus_extra": "HAR + basis + momentum + OI",
}
PATH_TITLES = {
    "ideal": "unlimited margin",
    "topup": "top-up from reserve",
    "liquidate": "no top-up",
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
    errors: list[float] | None = None,
) -> None:
    x = np.arange(len(values))
    ax.bar(x, values, width=0.62, color=color, edgecolor=p["surface"], linewidth=2, zorder=2)
    if errors is not None:
        ax.errorbar(
            x,
            values,
            yerr=errors,
            fmt="none",
            ecolor=p["text"],
            elinewidth=1.1,
            capsize=3,
            zorder=3,
        )
    spread = errors or [0.0] * len(values)
    tops = [v + e for v, e in zip(values, spread, strict=True)]
    bottoms = [v - e for v, e in zip(values, spread, strict=True)]
    high = max([0.0, *values, *tops])
    low = min([0.0, *values, *bottoms])
    pad = 0.14 * max(high - low, 1e-12)
    for xi, v, top, bottom in zip(x, values, tops, bottoms, strict=True):
        if v >= 0:
            y, va = max(top, v) + pad / 4, "bottom"
        else:
            y, va = min(bottom, v) - pad / 4, "top"
        ax.text(float(xi), y, fmt(v), ha="center", va=va, fontsize=8.2, color=p["text"])
    ax.set_xticks(x, labels)
    ax.axhline(0.0, color=p["muted"], linewidth=0.8, zorder=1)
    if high == low:  # every value is zero: give the empty panel a unit range
        high = 1.0
        pad = 0.0
    ax.set_ylim(low - pad if low < 0 else 0.0, high + pad if high > 0 else 0.0)


def _share_ylim(axes: Iterable[Axes]) -> None:
    """Put panels drawn one at a time on the union of their ranges. The panels must not be
    created with sharey, which would make every get_ylim return the last range set."""
    axes = list(axes)
    lows, highs = zip(*(ax.get_ylim() for ax in axes), strict=True)
    low, high = min(lows), max(highs)
    # Each panel padded its labels against its own range; pad again against the union.
    margin = 0.06 * (high - low)
    low, high = (low - margin if low < 0 else low), (high + margin if high > 0 else high)
    for i, ax in enumerate(axes):
        ax.set_ylim(low, high)
        if i:
            ax.tick_params(labelleft=False)


def _series(tables: Path) -> pd.DataFrame:
    return pd.read_csv(tables / "portfolio_series.csv", parse_dates=["time"]).set_index("time")


def headline(tables: Path, results: dict[str, Any], p: Palette) -> Figure:
    s = _series(tables)
    paths = results["headline"]["paths"]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.6), sharex=True)
    panels = [
        (
            "gross_funding",
            "Funding alone, compounded on the headline path",
            results["headline"]["gross_funding_annual"],
            p["secondary"],
        ),
        (
            "after_costs",
            "Funding less fees and spreads",
            results["headline"]["after_costs_annual"],
            p["muted"],
        ),
        (
            "net_topup",
            "Net: also basis moves, top-ups and liquidations",
            paths["topup"]["annual_return"],
            p["this_study"],
        ),
    ]
    for ax, (column, title, annual, color) in zip(axes, panels, strict=True):
        ax.plot(s.index, (s[column] - 1.0) * 100.0, color=color, linewidth=1.6)
        ax.axhline(0.0, color=p["muted"], linewidth=0.8)
        ax.set_title(f"{title}: {annual * 100:+.1f}% a year", loc="left")
        ax.set_ylabel("Cumulative, %")
    fig.tight_layout()
    return fig


def heatmap(tables: Path, p: Palette) -> Figure:
    h = pd.read_csv(tables / "heatmap.csv")
    wide = h.pivot_table(index="symbol", columns="month", values="annualized", aggfunc="first")
    wide = wide.loc[wide.mean(axis=1).sort_values(ascending=False).index]  # highest payer on top
    values = wide.to_numpy() * 100.0
    limit = float(np.nanpercentile(np.abs(values), 97)) or 1.0
    cmap = LinearSegmentedColormap.from_list(
        "funding", [p["benchmark"], p["grid"], p["this_study"]]
    ).with_extremes(bad=p["surface"])
    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.24 * len(wide) + 1.5)))
    image = ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
    )
    ax.set_yticks(range(len(wide)), wide.index, fontsize=7.5)
    step = max(1, len(wide.columns) // 12)
    ax.set_xticks(
        range(0, len(wide.columns), step), list(wide.columns)[::step], rotation=0, fontsize=7.5
    )
    ax.grid(visible=False)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    bar.set_label("Annualized funding, %")
    ax.set_title(
        f"Annualized funding by month: the {len(wide)} contracts with the longest history, "
        "highest average on top (blue: shorts are paid)",
        loc="left",
    )
    fig.tight_layout()
    return fig


def cross_section(tables: Path, p: Palette) -> Figure:
    c = pd.read_csv(tables / "cross_section.csv")
    x = np.arange(len(c))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True)
    median = c["median"].to_numpy() * 100.0
    axes[0].bar(
        x,
        median,
        width=0.8,
        color=np.where(median >= 0, p["this_study"], p["benchmark"]),
        edgecolor=p["surface"],
        linewidth=0.6,
        zorder=2,
    )
    axes[0].axhline(0.0, color=p["muted"], linewidth=0.8, zorder=1)
    axes[0].set_ylabel("% a year")
    axes[0].set_title(
        "Median contract: annualized funding each month (blue: shorts are paid)", loc="left"
    )
    share = c["share_negative"].to_numpy() * 100.0
    axes[1].bar(x, share, width=0.8, color=p["muted"], edgecolor=p["surface"], linewidth=0.6)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("% of contracts")
    axes[1].set_title("Share of contracts where funding was negative for the month", loc="left")
    january = [i for i, m in enumerate(c["month"]) if str(m).endswith("-01")]
    ticks = january or list(range(0, len(c), max(1, len(c) // 8)))
    axes[1].set_xticks(
        ticks, [str(c["month"].iloc[i])[:4] if january else str(c["month"].iloc[i]) for i in ticks]
    )
    fig.text(
        0.01,
        -0.02,
        f"Each month covers every contract with funding prints in it (between "
        f"{int(c['contracts'].min())} and {int(c['contracts'].max())} contracts).",
        fontsize=8,
        color=p["muted"],
    )
    fig.tight_layout()
    return fig


def forecast_skill(tables: Path, p: Palette) -> Figure:
    s = pd.read_csv(tables / "forecast_skill.csv")
    models = [m for m in ("har", "har_plus") if m in set(s["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.8), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        g = s.loc[s["model"] == model].sort_values("horizon")
        labels = [
            f"{h} period{'s' if h > 1 else ''}\nt = {t:.1f}"
            for h, t in zip(g["horizon"], g["dm_t"], strict=True)
        ]
        _bars(
            ax,
            labels=labels,
            values=list(g["r2_vs_persistence"] * 100.0),
            color=p["this_study"],
            p=p,
            fmt=lambda v: f"{v:+.0f}%",
        )
        ax.set_title(MODEL_TITLES[model], loc="left")
    axes[0][0].set_ylabel("Out-of-sample MSE reduction\nagainst persistence, %")
    _share_ylim(axes[0])
    fig.text(
        0.01,
        -0.03,
        "Horizon in 8-hour funding periods; t is the Diebold-Mariano statistic "
        "with an EWC long-run variance.",
        fontsize=8,
        color=p["muted"],
    )
    fig.tight_layout()
    return fig


def drawdown(tables: Path, config: Config, p: Palette) -> Figure:
    s = _series(tables)
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.fill_between(
        s.index, s["drawdown"] * 100.0, 0.0, color=p["this_study"], alpha=0.85, linewidth=0
    )
    low = float(s["drawdown"].min() * 100.0)
    for episode in config.episodes:
        start = pd.Timestamp(episode.start, tz="UTC")
        end = pd.Timestamp(episode.end, tz="UTC")
        if end < s.index[0] or start > s.index[-1]:
            continue
        # Matplotlib dates are days since its 1970-01-01 epoch.
        left, right = start.timestamp() / 86400.0, end.timestamp() / 86400.0
        ax.axvspan(left, right, color=p["highlight"], alpha=0.25, linewidth=0)
        ax.text(left, low * 1.08, episode.name, fontsize=8, color=p["text"], va="top")
    ax.set_ylim(low * 1.25 if low < 0 else -1.0, 0.5)
    ax.set_ylabel("Drawdown, %")
    ax.set_title("Portfolio drawdown with the stress episodes shaded", loc="left")
    fig.tight_layout()
    return fig


def rebuild(tables: Path, results: dict[str, Any], p: Palette) -> Figure | None:
    path = tables / "rebuild.csv"
    if not path.is_file():
        return None
    r = results["estimates"]["rebuild"]
    conventions = list(r["conventions"])
    fig, axes = plt.subplots(1, len(conventions) + 1, figsize=(4.4 * (len(conventions) + 1), 3.8))
    for ax, convention in zip(axes, conventions, strict=False):
        by = r["conventions"][convention]["by_interval"]
        keys = sorted(by, key=int)
        _bars(
            ax,
            labels=[f"{k}-hour\n{by[k]['n']} prints" for k in keys],
            values=[100.0 * by[k]["share"] for k in keys],
            color=p["this_study"],
            p=p,
            fmt=lambda v: f"{v:.0f}%",
        )
        ax.set_title(f"'{convention}' convention", loc="left")
        ax.set_ylim(0, 115)
    axes[0].set_ylabel(f"Prints within {results['settings']['rebuild_tolerance_bp']:.0f} bp, %")
    frame = pd.read_csv(path)
    err = (frame[r["best"]] - frame["reported"]).abs() * 1e4
    ax = axes[-1]
    ax.hist(
        np.clip(err, 0, 5), bins=np.linspace(0, 5, 26), color=p["secondary"], edgecolor=p["surface"]
    )
    label = f"'{r['best']}' convention" if r.get("decided") else "both conventions agree"
    ax.set_title(f"Absolute error, {label}", loc="left")
    ax.set_xlabel("Basis points (clipped at 5)")
    ax.set_ylabel("Prints")
    fig.tight_layout()
    return fig


def liquidations(results: dict[str, Any], p: Palette) -> Figure:
    liq = results["liquidations"]
    symbols = list(liq)
    fig, axes = plt.subplots(1, len(symbols), figsize=(5.2 * len(symbols), 3.6), squeeze=False)
    for ax, symbol in zip(axes[0], symbols, strict=True):
        keys = list(liq[symbol])
        _bars(
            ax,
            labels=[f"{k}x" for k in keys],
            values=[float(liq[symbol][k] or 0) for k in keys],
            color=p["benchmark"],
            p=p,
            fmt=lambda v: f"{v:.0f}",
        )
        ax.set_title(f"{symbol}: liquidations with no top-up", loc="left")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, max(2.0, ax.get_ylim()[1]))
        ax.set_xlabel("Leverage on the short perpetual")
    axes[0][0].set_ylabel("Liquidations over the sample")
    _share_ylim(axes[0])
    fig.tight_layout()
    return fig


def variants(tables: Path, results: dict[str, Any], p: Palette) -> Figure:
    v = pd.read_csv(tables / "portfolio_variants.csv").sort_values("sharpe").reset_index(drop=True)
    h = results["headline"]["portfolio"]
    head = (
        (v["model"] == h["model"])
        & (v["top_k"] == h["top_k"])
        & (v["min_volume"] == h["min_volume"])
        & (v["leverage"] == h["leverage"])
    )
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    colors = [p["benchmark"] if flag else p["this_study"] for flag in head]
    ax.bar(
        np.arange(len(v)),
        v["sharpe"],
        color=colors,
        edgecolor=p["surface"],
        linewidth=0.6,
        width=0.85,
    )
    ax.axhline(0.0, color=p["muted"], linewidth=0.8)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(v)} variants, sorted (orange: the headline variant)")
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title(
        "Every variant tried; deflated Sharpe of the headline: "
        f"{results['headline']['deflated_sharpe']:.2f}",
        loc="left",
    )
    fig.tight_layout()
    return fig


def venues(results: dict[str, Any], p: Palette) -> Figure | None:
    data = results["estimates"]["venues"]
    symbols = [s for s in data if "all" in data[s]]
    if not symbols:
        return None
    fig, axes = plt.subplots(1, len(symbols), figsize=(5.2 * len(symbols), 3.6), squeeze=False)
    for ax, symbol in zip(axes[0], symbols, strict=True):
        block = data[symbol]["all"]
        names = list(block["annualized"])
        _bars(
            ax,
            labels=[results["settings"]["venue_names"][n] for n in names],
            values=[100 * block["annualized"][n] for n in names],
            color=p["this_study"],
            p=p,
            fmt=lambda v: f"{v:+.1f}%",
        )
        ax.set_title(f"{symbol}, {block['first']} to {block['last']}", loc="left")
    axes[0][0].set_ylabel("Annualized funding, %")
    _share_ylim(axes[0])
    fig.tight_layout()
    return fig


def counterparty(tables: Path, results: dict[str, Any], p: Palette) -> Figure:
    c = pd.read_csv(tables / "counterparty.csv")
    lgds = sorted(c["lgd"].unique())
    carry = results["headline"]["portfolio"]["annual_return"] * 100.0
    fig, axes = plt.subplots(1, len(lgds), figsize=(4.2 * len(lgds), 3.6), sharey=True)
    for ax, lgd in zip(np.atleast_1d(axes), lgds, strict=True):
        g = c.loc[c["lgd"] == lgd].sort_values("hazard_scale")
        _bars(
            ax,
            labels=[f"{s:g}x" for s in g["hazard_scale"]],
            values=list(g["expected_loss"] * 100.0),
            color=p["benchmark"],
            p=p,
            fmt=lambda v: f"{v:.2f}%",
        )
        ax.axhline(carry, color=p["this_study"], linewidth=1.4)
        ax.set_title(f"Loss given failure {lgd:.0%}", loc="left")
        ax.set_xlabel("Hazard, multiple of the prior")
        top = max(carry, float(c["expected_loss"].max() * 100.0))
        ax.set_ylim(min(0.0, carry * 1.3), top * 1.3)
    np.atleast_1d(axes)[0].set_ylabel("Expected counterparty loss, % a year")
    fig.text(
        0.01,
        -0.03,
        "Blue rule: the portfolio's net annual return. Collateral split chosen "
        "to minimize expected shortfall.",
        fontsize=8,
        color=p["muted"],
    )
    fig.tight_layout()
    return fig


def benchmarks(results: dict[str, Any], config: Config, p: Palette) -> Figure:
    b = results["estimates"]["portfolio"]["benchmarks"]
    h = results["headline"]["portfolio"]
    quote = config.universe.quote
    bases = {s: s.removesuffix(quote) for s in config.universe.core_symbols}
    items = [("Carry\nportfolio", h)]
    for s, base in bases.items():
        if f"{s}_carry" in b:
            items.append((f"{base}\ncarry", b[f"{s}_carry"]))
    if "core_mix_carry" in b:
        items.append((" + ".join(bases.values()) + "\ncarry", b["core_mix_carry"]))
    for s, base in bases.items():
        if f"{s}_spot" in b:
            items.append((f"{base}\nspot", b[f"{s}_spot"]))
    labels = [k for k, _ in items]
    colors = [p["this_study"]] + [p["muted"]] * (len(items) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    _bars(
        axes[0],
        labels=labels,
        values=[100 * v["annual_return"] for _, v in items],
        color=colors,
        p=p,
        fmt=lambda x: f"{x:+.1f}%",
    )
    axes[0].set_title("Annual return, %", loc="left")
    _bars(
        axes[1],
        labels=labels,
        values=[v["sharpe"] for _, v in items],
        color=colors,
        p=p,
        fmt=lambda x: f"{x:.2f}",
        errors=[1.96 * v["sharpe_se"] for _, v in items],
    )
    axes[1].set_title("Sharpe ratio with 95% EWC intervals", loc="left")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.8)
    fig.tight_layout()
    return fig


def draw_all(
    *, config: Config, results: dict[str, Any], tables: Path, out: Path
) -> dict[str, dict[str, str]]:
    p = config.plot.palette
    _style(p)
    drawn: dict[str, Figure | None] = {
        "headline": headline(tables, results, p),
        "cross_section": cross_section(tables, p),
        "heatmap": heatmap(tables, p),
        "forecast": forecast_skill(tables, p),
        "drawdown": drawdown(tables, config, p),
        "rebuild": rebuild(tables, results, p),
        "liquidations": liquidations(results, p),
        "variants": variants(tables, results, p),
        "venues": venues(results, p),
        "counterparty": counterparty(tables, results, p),
        "benchmarks": benchmarks(results, config, p),
    }
    return {
        name: _save(fig, out=out, name=name, dpi=config.plot.dpi)
        for name, fig in drawn.items()
        if fig is not None
    }
