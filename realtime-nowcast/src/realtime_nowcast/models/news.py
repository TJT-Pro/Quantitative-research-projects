"""Nowcasts from fixed parameters, and the decomposition of each change into named causes.

Between two days with the same parameters, the change in the nowcast of quarter q splits as
in Banbura and Modugno (2010):

    y_new - y_old = (y_rev - y_old)                         revisions to data already out
                  + sum_j w_j (x_j - E[x_j | data_rev])     news in each new release

where data_rev is the new vintage with the new releases taken out. With fixed parameters
and a fixed pattern of missing data, the smoothed nowcast is linear in the data, so w_j is
the exact derivative of y_new with respect to x_j (kalman.impulse), and the identity holds
to rounding: `Change.residual` measures it and the pipeline stops if it is not tiny.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from realtime_nowcast.models import kalman
from realtime_nowcast.models.dfm import Params

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Fit:
    """The filter run on one panel with one parameter set."""

    params: Params
    filtered: kalman.Filtered
    xs: Array  # (T, N) standardized panel

    def smoothed(self, down_to: int) -> Array:
        return kalman.smoothed_means(self.filtered, self.params.a, down_to=down_to)


def fit(x: Array, params: Params, periods: pd.PeriodIndex) -> Fit:
    """`x` (T, N) in transformed units, rows on `periods`."""
    xs = (x - params.mx) / params.wx
    f = kalman.run_filter(
        xs.T,
        a=params.a,
        c=params.c,
        q=params.q,
        r=params.r,
        z0=params.z0,
        v0=params.v0,
        scale=params.pandemic.scale(periods),
    )
    return Fit(params=params, filtered=f, xs=xs)


def value_at(fitted: Fit, *, series: int, row: int, smoothed: Array | None = None) -> float:
    """The model's estimate of series `series` in data row `row` (0-based), transformed units."""
    zs = fitted.smoothed(row + 1) if smoothed is None else smoothed
    p = fitted.params
    return float(p.wx[series] * (p.c[series] @ zs[:, row + 1]) + p.mx[series])


@dataclass(frozen=True)
class Release:
    series: int
    row: int
    actual: float
    forecast: float
    weight: float  # change in the nowcast per unit of the datum (transformed units)

    @property
    def impact(self) -> float:
        return self.weight * (self.actual - self.forecast)


@dataclass(frozen=True)
class Change:
    old: float
    revised: float
    new: float
    releases: tuple[Release, ...]

    @property
    def revisions(self) -> float:
        return self.revised - self.old

    @property
    def news(self) -> float:
        return float(sum(r.impact for r in self.releases))

    @property
    def residual(self) -> float:
        return self.new - self.old - self.revisions - self.news


def new_cells(x_old: Array, x_new: Array) -> NDArray[np.int64]:
    """(row, series) of data missing before and present now."""
    return np.argwhere(np.isnan(x_old) & ~np.isnan(x_new))


def decompose(
    *, old: float, rev: Fit, new: Fit, x_old: Array, x_new: Array, target: int, row: int
) -> Change:
    """`x_old` is the earlier panel padded to the rows of `x_new`; `target` the series and
    `row` the data row of the nowcast."""
    cells = new_cells(x_old, x_new)
    low = int(min([row, *cells[:, 0].tolist()])) if cells.size else row
    zs_rev = rev.smoothed(low + 1)
    zs_new = new.smoothed(row + 1)
    p = new.params
    y_rev = value_at(rev, series=target, row=row, smoothed=zs_rev)
    y_new = value_at(new, series=target, row=row, smoothed=zs_new)
    releases = []
    for t, i in cells:
        dz = kalman.impulse(
            new.filtered, a=p.a, c=p.c, series=int(i), column=int(t) + 1, target=row + 1
        )
        weight = float(p.wx[target] * (p.c[target] @ dz) / p.wx[i])
        releases.append(
            Release(
                series=int(i),
                row=int(t),
                actual=float(x_new[t, i]),
                forecast=value_at(rev, series=int(i), row=int(t), smoothed=zs_rev),
                weight=weight,
            )
        )
    return Change(old=old, revised=y_rev, new=y_new, releases=tuple(releases))
