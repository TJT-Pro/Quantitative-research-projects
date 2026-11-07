"""One function per published figure. One series per panel: side-by-side panels and bars,
never several lines sharing an axis. Bars carry direct value labels."""

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

from realtime_nowcast.config import Config

Palette = dict[str, str]


def _signed(value: float) -> str:
    """A signed label with two significant digits, so deciles near zero stay distinct."""
    if value == 0:
        return "0"
    text = f"{value:+.2g}"
    return text.replace("e-0", "e-")


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
    bottoms: list[float] | None = None,
    texts: list[str] | None = None,
) -> None:
    x = np.arange(len(values))
    base = np.zeros(len(values)) if bottoms is None else np.asarray(bottoms)
    ax.bar(
        x,
        values,
        bottom=base,
        width=0.62,
        color=color,
        edgecolor=p["surface"],
        linewidth=2,
        zorder=2,
    )
    tops = base + np.asarray(values)
    high = max([0.0, *tops.tolist(), *base.tolist()])
    low = min([0.0, *tops.tolist(), *base.tolist()])
    pad = 0.14 * max(high - low, 1e-12)
    shown = texts if texts is not None else [fmt(v) for v in values]
    for xi, v, top, text in zip(x, values, tops, shown, strict=True):
        y, va = (top + pad / 4, "bottom") if v >= 0 else (top - pad / 4, "top")
        ax.text(float(xi), y, text, ha="center", va=va, fontsize=8.2, color=p["text"])
    ax.set_xticks(x, labels)
    ax.axhline(0.0, color=p["muted"], linewidth=0.8, zorder=1)
    ax.set_ylim(low - pad if low < 0 else 0.0, high + pad if high > 0 else 1.0)


def _share_ylim(axes: Iterable[Axes]) -> None:
    """Put panels drawn one at a time on the union of their ranges."""
    axes = list(axes)
    lows, highs = zip(*(ax.get_ylim() for ax in axes), strict=True)
    low, high = min(lows), max(highs)
    for i, ax in enumerate(axes):
        ax.set_ylim(low, high)
        if i:
            ax.tick_params(labelleft=False)


def rmse_by_horizon(results: dict[str, Any], p: Palette) -> Figure:
    groups = results["estimates"]["evaluation"]["groups"]
    labels = [g["group"] for g in groups][::-1]
    model = [g["all"]["model_rmse"] for g in groups][::-1]
    bench = [g["all"]["gdpnow_rmse"] for g in groups][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, values, color, title in (
        (axes[0], model, p["this_study"], "This model"),
        (axes[1], bench, p["benchmark"], "GDPNow"),
    ):
        _bars(ax, labels=labels, values=values, color=color, p=p, fmt=lambda v: f"{v:.2f}")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Days before the advance estimate")
    axes[0].set_ylabel("RMSE against the advance estimate (pp)")
    _share_ylim(axes)
    fig.tight_layout()
    return fig


def nowcast_path(tables: Path, results: dict[str, Any], p: Palette) -> Figure:
    est = results["estimates"]
    quarter, advance = est["example_quarter"], est["example_advance"]
    release = pd.Timestamp(est["example_release"])
    model = pd.read_csv(tables / "path_model.csv", parse_dates=["date"]).sort_values("date")
    bench = pd.read_csv(tables / "path_gdpnow.csv", parse_dates=["date"]).sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True)
    for ax, frame, color, title in (
        (
            axes[0],
            model,
            p["this_study"],
            f"This model's nowcast of {quarter}; the diamond is the advance estimate "
            f"({advance:.1f}%)",
        ),
        (
            axes[1],
            bench,
            p["benchmark"],
            f"GDPNow's nowcast of {quarter}; the diamond is the advance estimate",
        ),
    ):
        ax.step(frame["date"], frame["nowcast"], where="post", color=color, linewidth=1.8)
        ax.plot([release], [advance], marker="D", color=p["text"], markersize=6, linestyle="none")
        ax.set_title(title, loc="left")
        ax.set_ylabel("Annualized growth (%)")
    _share_ylim_values(axes, [*model["nowcast"], *bench["nowcast"], advance])
    fig.tight_layout()
    return fig


def _share_ylim_values(axes: Iterable[Axes], values: list[float]) -> None:
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    pad = 0.12 * max(high - low, 0.5)
    for ax in axes:
        ax.set_ylim(low - pad, high + pad)


def waterfall(tables: Path, results: dict[str, Any], p: Palette) -> Figure:
    w = pd.read_csv(tables / "waterfall.csv")
    labels, values, bottoms, colors, texts = [], [], [], [], []
    running = 0.0
    for step, kind, value in zip(
        w["step"].astype(str), w["kind"].astype(str), w["value"].to_numpy(dtype=float), strict=True
    ):
        labels.append(
            step.replace(" and ", " &\n").replace(" ", "\n", 1) if len(step) > 12 else step
        )
        values.append(float(value))
        texts.append(f"{value:.2f}" if kind == "level" else f"{value:+.2f}")
        if kind == "level":
            bottoms.append(0.0)
            colors.append(p["muted"])
            running = float(value)
        else:
            bottoms.append(running)
            colors.append(p["secondary"] if value >= 0 else p["benchmark"])
            running += float(value)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    _bars(
        ax,
        labels=labels,
        values=values,
        color=colors,
        p=p,
        fmt=lambda v: f"{v:+.2f}",
        bottoms=bottoms,
        texts=texts,
    )
    ax.set_title(f"What moved the nowcast of {results['estimates']['example_quarter']}", loc="left")
    ax.set_ylabel("Annualized growth (%) and changes (pp)")
    ax.tick_params(axis="x", labelsize=7.6)
    fig.tight_layout()
    return fig


def yield_reaction(results: dict[str, Any], config: Config, p: Palette) -> Figure:
    tests = results["estimates"]["market"]
    fig, axes = plt.subplots(1, len(config.market.yields), figsize=(10.5, 3.9))
    axes_list = list(np.atleast_1d(axes))
    for ax, series in zip(axes_list, config.market.yields, strict=True):
        t = tests[series]
        d = t["deciles"]
        x = np.arange(len(d))
        means = [b["change"] for b in d]
        half = [b["half_width"] for b in d]
        color = p["this_study"] if series == config.market.primary else p["secondary"]
        ax.bar(x, means, width=0.62, color=color, edgecolor=p["surface"], linewidth=2, zorder=2)
        ax.errorbar(
            x, means, yerr=half, fmt="none", ecolor=p["text"], elinewidth=1, capsize=3, zorder=3
        )
        ax.axhline(0.0, color=p["muted"], linewidth=0.8, zorder=1)
        ax.set_xticks(x, [_signed(b["news"]) for b in d], rotation=45)
        a = t["all"]
        ax.set_title(
            f"{series}: {a['slope']:.2f} bp per pp of news [{a['ci_low']:.2f}, {a['ci_high']:.2f}]",
            loc="left",
        )
        ax.set_xlabel("Release-day news, decile mean (pp of GDP growth)")
    axes_list[0].set_ylabel("Mean yield change that day (bp)")
    _share_ylim(axes_list)
    fig.tight_layout()
    return fig


def errors_by_quarter(tables: Path, p: Palette) -> Figure:
    binned = pd.read_csv(tables / "binned.csv")
    last = binned.sort_values("horizon").groupby("quarter").head(1).sort_values("quarter")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.4), sharex=True)
    x = np.arange(len(last))
    for ax, column, color, title in (
        (
            axes[0],
            "model_error",
            p["this_study"],
            "This model: last nowcast minus the advance estimate",
        ),
        (
            axes[1],
            "gdpnow_error",
            p["benchmark"],
            "GDPNow: last nowcast minus the advance estimate",
        ),
    ):
        ax.bar(
            x, last[column], width=0.7, color=color, edgecolor=p["surface"], linewidth=1, zorder=2
        )
        ax.axhline(0.0, color=p["muted"], linewidth=0.8, zorder=1)
        ax.set_title(title, loc="left")
        ax.set_ylabel("pp")
    step = max(1, len(last) // 12)
    axes[1].set_xticks(x[::step], list(last["quarter"])[::step], rotation=45)
    _share_ylim(axes)
    for ax in axes:
        ax.tick_params(labelleft=True)
    fig.tight_layout()
    return fig


def draw_all(
    *, config: Config, results: dict[str, Any], tables: Path, out: Path
) -> dict[str, dict[str, str]]:
    p = config.plot.palette
    _style(p)
    drawn: dict[str, Figure] = {
        "rmse": rmse_by_horizon(results, p),
        "path": nowcast_path(tables, results, p),
        "waterfall": waterfall(tables, results, p),
        "yields": yield_reaction(results, config, p),
        "errors": errors_by_quarter(tables, p),
    }
    return {
        name: _save(fig, out=out, name=name, dpi=config.plot.dpi) for name, fig in drawn.items()
    }
