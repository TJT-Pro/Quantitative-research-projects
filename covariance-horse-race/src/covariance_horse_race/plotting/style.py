"""Shared figure style.

Models are colored by estimator family, in a fixed slot order that passes the
colorblind-separation checks (validated categorical palette, slots 1-4), and
distinguished within a family by line style and marker. Color therefore follows
the model, never its rank, and identity never rests on color alone: every line
is also direct-labeled or marker-coded, and every figure has a table twin in
reports/tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # backend is chosen before pyplot loads
from matplotlib.figure import Figure

__all__ = ["FAMILY_COLORS", "ModelStyle", "apply", "model_styles", "plt", "save"]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

FAMILY_COLORS = {
    "unconditional": "#2a78d6",
    "shrinkage": "#eb6834",
    "factor": "#1baf7a",
    "conditional": "#eda100",
}
FAMILY_OF_KIND = {
    "sample": "unconditional",
    "ewma": "unconditional",
    "linear_shrinkage": "shrinkage",
    "nonlinear_shrinkage": "shrinkage",
    "observable_factor": "factor",
    "statistical_factor": "factor",
    "dcc": "conditional",
}
_LINESTYLES = ["-", "--", ":"]
_MARKERS = ["o", "s", "^"]

# Sequential blue ramp (light to dark) for heatmaps.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


@dataclass(frozen=True)
class ModelStyle:
    label: str
    color: str
    linestyle: str
    marker: str


def model_styles(models: list[tuple[str, str, str]]) -> dict[str, ModelStyle]:
    """Map (name, kind, label) triples, in config order, to a stable style."""
    seen: dict[str, int] = {}
    styles: dict[str, ModelStyle] = {}
    for name, kind, label in models:
        family = FAMILY_OF_KIND[kind]
        position = seen.get(family, 0)
        seen[family] = position + 1
        styles[name] = ModelStyle(
            label=label,
            color=FAMILY_COLORS[family],
            linestyle=_LINESTYLES[position % len(_LINESTYLES)],
            marker=_MARKERS[position % len(_MARKERS)],
        )
    return styles


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
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
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
        }
    )


def save(fig: Figure, directory: Path, stem: str, dpi: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{stem}.png", directory / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(paths[1], bbox_inches="tight", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    return paths
