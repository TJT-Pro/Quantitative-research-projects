"""Gap filling used only to start EM (a port of remNaNs_spline in the New York Fed code).

EM itself handles missing values exactly through the Kalman smoother; these fills give the
principal components and regressions that set the starting parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]

MA_HALF_WIDTH = 3  # the k = 3 centred moving average of the reference code
SPARSE_ROW_SHARE = 0.8  # leading and closing rows with more than 80% missing are dropped


@dataclass(frozen=True)
class Filled:
    filled: Array  # rows kept, every gap filled
    missing: Mask  # where the kept rows were missing before filling
    kept: Mask  # which input rows were kept


def _edge_rows(sparse: Mask) -> Mask:
    """Leading and closing runs of `sparse` rows."""
    n = sparse.size
    lead = np.cumsum(sparse) == np.arange(1, n + 1)
    end = (np.cumsum(sparse[::-1]) == np.arange(1, n + 1))[::-1]
    return np.asarray(lead | end, dtype=np.bool_)


def _moving_average(x: Array, half: int) -> Array:
    padded = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    kernel = np.full(2 * half + 1, 1.0 / (2 * half + 1))
    return np.asarray(np.convolve(padded, kernel, mode="valid"), dtype=np.float64)


def _fill_column(x: Array, half: int) -> Array:
    x = x.copy()
    seen = np.flatnonzero(~np.isnan(x))
    if seen.size == 0:
        raise ValueError("a series has no observations at all")
    t1, t2 = seen[0], seen[-1]
    if seen.size > 1:
        span = np.arange(t1, t2 + 1)
        x[span] = CubicSpline(seen, x[seen], bc_type="not-a-knot")(span)
    gaps = np.isnan(x)
    x[gaps] = np.median(x[~gaps])
    smooth = _moving_average(x, half)
    x[gaps] = smooth[gaps]
    return x


def fill_for_start(x: Array, *, half: int = MA_HALF_WIDTH) -> Filled:
    """Drop leading and closing rows that are mostly missing, spline-interpolate inside each
    series, and put a moving average of the median-filled series at its ends."""
    missing = np.isnan(x)
    kept = ~_edge_rows(missing.sum(axis=1) > x.shape[1] * SPARSE_ROW_SHARE)
    trimmed = x[kept]
    values = np.column_stack([_fill_column(trimmed[:, i], half) for i in range(trimmed.shape[1])])
    return Filled(filled=values, missing=np.isnan(trimmed), kept=kept)


def drop_empty_edges(x: Array) -> Array:
    """Drop leading and closing rows with no observation at all."""
    return x[~_edge_rows(np.isnan(x).all(axis=1))]
