"""One function per published figure. One series per panel: bars, dots with intervals
and small multiples, never several lines sharing an axis. Every bar carries a direct
value label, because two palette colors fall below 3:1 contrast on the surface."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from factor_zoo_trial.config import Config

Palette = dict[str, str]
PERIOD_LABELS = ("In sample", "After sample,\nbefore publication", "After publication")
PROCEDURE_ORDER = ("unadjusted", "bonferroni", "holm", "bhy", "hlz", "romano_wolf")
PROCEDURE_TITLES = {
    "unadjusted": "Unadjusted, one-sided 5%",
    "bonferroni": "Bonferroni",
    "holm": "Holm",
    "bh": "Benjamini-Hochberg",
    "bhy": "Benjamini-Hochberg-Yekutieli",
    "hlz": "Harvey-Liu-Zhu, t > 3",
    "romano_wolf": "Romano-Wolf, whole-month bootstrap",
}
VARIANT_TITLES = {
    "original": "As in the papers",
    "value_weighted": "Value-weighted deciles",
    "nyse_only": "NYSE stocks only",
    "price_gt_5": "Price above $5",
    "me_gt_nyse20": "No microcaps",
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
    # No timestamps in the metadata, so an unchanged figure is an unchanged file.
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return {"png": f"reports/figures/{name}.png", "svg": f"reports/figures/{name}.svg"}


def _bars_with_ci(
    ax: Axes,
    *,
    labels: list[str],
    values: list[float],
    lows: list[float] | None,
    highs: list[float] | None,
    color: str,
    p: Palette,
    fmt: Callable[[float], str],
) -> None:
    x = np.arange(len(values))
    ax.bar(x, values, width=0.62, color=color, edgecolor=p["surface"], linewidth=2, zorder=2)
    if lows is not None and highs is not None:
        err = np.array(
            [
                [v - lo for v, lo in zip(values, lows, strict=True)],
                [hi - v for v, hi in zip(values, highs, strict=True)],
            ]
        )
        # A published point estimate has no interval: draw whiskers only where one exists.
        has = err.sum(axis=0) > 0
        if has.any():
            ax.errorbar(
                x[has],
                np.asarray(values)[has],
                yerr=err[:, has],
                fmt="none",
                ecolor=p["text"],
                elinewidth=1.2,
                capsize=4,
                zorder=3,
            )
    tops = highs if highs is not None else values
    span = max(abs(v) for v in [*values, *tops, 1e-9])
    for xi, v, top in zip(x, values, tops, strict=True):
        y = max(top, v, 0.0) + 0.04 * span
        ax.text(float(xi), y, fmt(v), ha="center", va="bottom", fontsize=8.5, color=p["text"])
    ax.set_xticks(x, labels)
    ax.axhline(0.0, color=p["muted"], linewidth=0.8, zorder=1)
    ax.set_ylim(min(0.0, min([*values, *(lows or values)]) * 1.15), span * 1.22)


def headline(results: dict[str, Any], p: Palette) -> Figure:
    h = results["headline"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, key, title in (
        (axes[0], "decay", f"All predictors (n = {h['decay']['n_signals']})"),
        (axes[1], "decay_clear", f"Clear replications only (n = {h['decay_clear']['n_signals']})"),
    ):
        d = h[key]
        periods = ("in_sample", "post_sample", "post_publication")
        _bars_with_ci(
            ax,
            labels=list(PERIOD_LABELS),
            values=[d[k]["value"] for k in periods],
            lows=[d[k]["low"] for k in periods],
            highs=[d[k]["high"] for k in periods],
            color=p["this_study"],
            p=p,
            fmt=lambda v: f"{v:.2f}%",
        )
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Average long-short return, % per month")
    fig.text(
        0.01,
        -0.04,
        "Bars: average predictor fixed effect plus the period shift. "
        "Whiskers: 95% intervals, standard errors clustered by month.",
        fontsize=8,
        color=p["muted"],
    )
    fig.tight_layout()
    return fig


def discoveries(flags: pd.DataFrame, results: dict[str, Any], p: Palette) -> Figure:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True, sharey=True)
    t = flags["t"].clip(-4, 12).to_numpy()
    bins = np.arange(-4.0, 12.5, 0.5)
    n = len(flags)
    for ax, name in zip(axes.ravel(), PROCEDURE_ORDER, strict=True):
        found = flags[name].to_numpy(dtype=bool)
        kept, _ = np.histogram(t[found], bins=bins)
        rest, _ = np.histogram(t[~found], bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(centers, rest, width=0.46, color=p["grid"], edgecolor=p["surface"], linewidth=1)
        ax.bar(
            centers,
            kept,
            width=0.46,
            bottom=rest,
            color=p["this_study"],
            edgecolor=p["surface"],
            linewidth=1,
        )
        ax.set_title(
            f"{PROCEDURE_TITLES[name]}: {int(found.sum())} of {n}", loc="left", fontsize=9.5
        )
    for ax in axes[1]:
        ax.set_xlabel("In-sample t-statistic (EWC standard error)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predictors")
    handles = [
        mpl.patches.Patch(color=p["this_study"], label="Discovery"),
        mpl.patches.Patch(color=p["grid"], label="Not a discovery"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def regions(results: dict[str, Any], p: Palette) -> Figure:
    regs = results["estimates"]["jkp"]["regions"]
    names = list(regs)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    _bars_with_ci(
        ax,
        labels=[
            f"{k.upper() if len(k) == 3 else k.capitalize()}\n{regs[k]['n_counted']} factors"
            for k in names
        ],
        values=[100 * regs[k]["replication_rate"] for k in names],
        lows=[100 * regs[k]["replication_low"] for k in names],
        highs=[100 * regs[k]["replication_high"] for k in names],
        color=p["this_study"],
        p=p,
        fmt=lambda v: f"{v:.0f}%",
    )
    ax.set_ylabel("Replication rate, %")
    ax.set_title(
        "Share of originally significant factors with a 95% posterior interval above zero",
        loc="left",
    )
    fig.text(
        0.01,
        -0.06,
        "Whiskers: 95% posterior interval of the rate (JKP simulation from the joint posterior).",
        fontsize=8,
        color=p["muted"],
    )
    fig.tight_layout()
    return fig


def clusters(posterior: pd.DataFrame, p: Palette) -> Figure:
    names = sorted(posterior["cluster"].unique())
    cols = 5
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.6 * rows), sharex=True)
    lo = float(posterior["low_post"].min())
    hi = float(posterior["high_post"].max())
    for ax, name in zip(axes.ravel(), names, strict=False):
        g = posterior.loc[posterior["cluster"] == name].sort_values("alpha_post")
        y = np.arange(len(g))
        ax.errorbar(
            g["alpha_post"],
            y,
            xerr=[g["alpha_post"] - g["low_post"], g["high_post"] - g["alpha_post"]],
            fmt="o",
            color=p["this_study"],
            ecolor=p["this_study"],
            markersize=4,
            elinewidth=1.1,
            capsize=0,
        )
        ax.axvline(0.0, color=p["muted"], linewidth=0.8)
        ax.set_yticks([])
        ax.grid(axis="x", color=p["grid"])
        ax.grid(axis="y", visible=False)
        share = float((g["low_post"] > 0).mean())
        ax.set_title(
            f"{name}\n{len(g)} factors, {share:.0%} significant",
            loc="left",
            fontsize=9,
        )
        ax.set_xlim(lo - 0.05, hi + 0.05)
    for ax in axes.ravel()[len(names) :]:
        ax.set_visible(False)
    flat = axes.ravel()
    for i, ax in enumerate(flat[: len(names)]):
        # Label the x axis on the lowest visible panel of each column.
        if i + cols >= len(names):
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("US posterior alpha, % per month")
    fig.tight_layout()
    return fig


def shifts(results: dict[str, Any], p: Palette) -> Figure:
    sh = results["shifts"]
    keys = list(sh)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, field, title, color in (
        (axes[0], "share_post_sample", "After the sample, before publication", p["secondary"]),
        (axes[1], "share_post_publication", "After publication", p["this_study"]),
    ):
        _bars_with_ci(
            ax,
            labels=[f"{k} yr" for k in keys],
            values=[100 * sh[k][field]["value"] for k in keys],
            lows=[100 * sh[k][field]["low"] for k in keys],
            highs=[100 * sh[k][field]["high"] for k in keys],
            color=color,
            p=p,
            fmt=lambda v: f"{v:+.0f}%",
        )
        ax.set_title(title, loc="left")
        ax.set_xlabel("Publication date moved back by")
    axes[0].set_ylabel("Change in return, % of in-sample mean")
    fig.tight_layout()
    return fig


def tradability(results: dict[str, Any], p: Palette) -> Figure:
    v = results["variants"]
    names = list(v)
    labels = [VARIANT_TITLES.get(k, k).replace(" ", "\n", 1) for k in names]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    _bars_with_ci(
        axes[0],
        labels=labels,
        values=[v[k]["decay"]["in_sample"]["value"] for k in names],
        lows=[v[k]["decay"]["in_sample"]["low"] for k in names],
        highs=[v[k]["decay"]["in_sample"]["high"] for k in names],
        color=p["this_study"],
        p=p,
        fmt=lambda x: f"{x:.2f}%",
    )
    axes[0].set_title("In-sample return, % per month", loc="left")
    _bars_with_ci(
        axes[1],
        labels=labels,
        values=[100 * v[k]["decay"]["share_post_publication"]["value"] for k in names],
        lows=[100 * v[k]["decay"]["share_post_publication"]["low"] for k in names],
        highs=[100 * v[k]["decay"]["share_post_publication"]["high"] for k in names],
        color=p["benchmark"],
        p=p,
        fmt=lambda x: f"{x:+.0f}%",
    )
    axes[1].set_title("Post-publication change, % of in sample", loc="left")
    _bars_with_ci(
        axes[2],
        labels=labels,
        values=[float(v[k]["testing"]["counts"]["bhy"]) for k in names],
        lows=None,
        highs=None,
        color=p["secondary"],
        p=p,
        fmt=lambda x: f"{x:.0f}",
    )
    axes[2].set_title("Discoveries under BHY", loc="left")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    return fig


def space(scores: pd.DataFrame, p: Palette) -> Figure:
    frame = scores.sort_values("gain_vs_no_skill")
    height = max(3.5, 0.17 * len(frame) + 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(11, height), sharey=True)
    y = np.arange(len(frame))
    for ax, field, title, color in (
        (axes[0], "gain_vs_no_skill", "Against no skill (true alphas zero)", p["this_study"]),
        (
            axes[1],
            "gain_vs_dispersion",
            "Against the US spread with no US direction",
            p["benchmark"],
        ),
    ):
        ax.barh(y, frame[field], color=color, height=0.7, edgecolor=p["surface"], linewidth=1)
        ax.axvline(0.0, color=p["muted"], linewidth=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Log predictive score gain per factor")
        ax.grid(axis="x", color=p["grid"])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [c.upper() for c in frame["location"]], fontsize=7)
    fig.tight_layout()
    return fig


def simulation(results: dict[str, Any], p: Palette, level: float) -> Figure:
    testing = results["estimates"]["simulation"]["testing"]
    shares = list(testing)
    procs = ["bonferroni", "holm", "romano_wolf", "bh", "bhy"]
    fig, axes = plt.subplots(2, len(shares), figsize=(4.0 * len(shares), 6.4), sharey="row")
    axes = np.atleast_2d(axes)
    short = {"bonferroni": "Bonf.", "holm": "Holm", "romano_wolf": "RW", "bh": "BH", "bhy": "BHY"}
    for j, share in enumerate(shares):
        block = testing[share]
        realized = [100 * block[k]["realized"] for k in procs]
        x = np.arange(len(procs))
        ax = axes[0, j]
        ax.bar(
            x,
            realized,
            color=[p["this_study"]] * 3 + [p["secondary"]] * 2,
            edgecolor=p["surface"],
            linewidth=2,
            width=0.62,
        )
        ax.axhline(100 * level, color=p["benchmark"], linewidth=1.4)
        for xi, v in zip(x, realized, strict=True):
            # A surface-colored box keeps the number readable where it meets the rule.
            ax.text(
                float(xi),
                v + 0.3,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=4,
                bbox={"facecolor": p["surface"], "edgecolor": "none", "pad": 1.0},
            )
        ax.set_xticks(x, [short[k] for k in procs])
        ax.set_title(f"{float(share):.0%} of alphas zero", loc="left")
        power = [100 * block[k]["power"] for k in procs]
        ax2 = axes[1, j]
        ax2.bar(x, power, color=p["highlight"], edgecolor=p["surface"], linewidth=2, width=0.62)
        for xi, v in zip(x, power, strict=True):
            ax2.text(float(xi), v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(x, [short[k] for k in procs])
        ax2.set_ylim(0, 110)
    axes[0, 0].set_ylabel("Realized error rate, %\n(FWER: Bonf., Holm, RW; FDR: BH, BHY)")
    axes[1, 0].set_ylabel("Power, % of true effects found")
    top = max(100 * level * 1.8, *(100 * testing[s][k]["realized"] for s in shares for k in procs))
    axes[0, 0].set_ylim(0, top * 1.2)
    fig.text(0.01, -0.02, f"Orange rule: the {level:.0%} target.", fontsize=8, color=p["muted"])
    fig.tight_layout()
    return fig


def benchmarks(results: dict[str, Any], p: Palette) -> Figure:
    mp = results["estimates"]["osap"]["mclean_pontiff"]
    jkp = results["estimates"]["jkp"].get("benchmark")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, field, title in (
        (axes[0], "share_post_sample", "McLean-Pontiff: after the sample"),
        (axes[1], "share_post_publication", "McLean-Pontiff: after publication"),
    ):
        est = mp["estimate"][field]
        _bars_with_ci(
            ax,
            labels=["This study", "Published"],
            values=[100 * est["value"], 100 * mp["published"][field]],
            lows=[100 * est["low"], 100 * mp["published"][field]],
            highs=[100 * est["high"], 100 * mp["published"][field]],
            color=p["this_study"],
            p=p,
            fmt=lambda v: f"{v:+.0f}%",
        )
        ax.patches[1].set_facecolor(p["benchmark"])
        ax.set_title(title, loc="left", fontsize=9.5)
    axes[0].set_ylabel("Change, % of in-sample mean")
    if jkp is not None:
        e = jkp["estimate"]
        _bars_with_ci(
            axes[2],
            labels=["This study", "Published"],
            values=[100 * e["replication_rate"], 100 * jkp["published"]["replication_rate"]],
            lows=[100 * e["replication_low"], 100 * jkp["published"]["replication_rate"]],
            highs=[100 * e["replication_high"], 100 * jkp["published"]["replication_rate"]],
            color=p["this_study"],
            p=p,
            fmt=lambda v: f"{v:.0f}%",
        )
        axes[2].patches[1].set_facecolor(p["benchmark"])
        axes[2].set_title("JKP: US replication rate", loc="left", fontsize=9.5)
        axes[2].set_ylabel("%")
    fig.tight_layout()
    return fig


def draw_all(
    *, config: Config, results: dict[str, Any], tables: Path, out: Path
) -> dict[str, dict[str, str]]:
    p = config.plot.palette
    _style(p)
    dpi = config.plot.dpi
    source = config.space.source_region
    drawn = {
        "headline": headline(results, p),
        "discoveries": discoveries(pd.read_csv(tables / "discoveries.csv"), results, p),
        "regions": regions(results, p),
        "clusters": clusters(pd.read_csv(tables / f"posterior_{source}.csv"), p),
        "shifts": shifts(results, p),
        "tradability": tradability(results, p),
        "space": space(pd.read_csv(tables / "space_scores.csv"), p),
        "simulation": simulation(results, p, config.multiple_testing.level),
        "benchmarks": benchmarks(results, p),
    }
    return {name: _save(fig, out=out, name=name, dpi=dpi) for name, fig in drawn.items()}
