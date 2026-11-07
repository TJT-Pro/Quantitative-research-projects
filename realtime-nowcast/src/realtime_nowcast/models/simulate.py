"""Panels drawn from a known factor model, for the recovery tests and the fixture.

The parameters have the layout dfm.estimate produces (models.dfm.Structure), so a panel
drawn here can be estimated and its factor and parameters compared with the truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from realtime_nowcast.models.dfm import LAGS, TENT, Structure

Array = NDArray[np.float64]

GLOBAL_AR = 0.8
BLOCK_AR = 0.5
IDIO_AR = 0.2
IDIO_VARIANCE = 0.4
NOISE_SD = 1e-2  # the model's measurement error is 1e-4 in variance


@dataclass(frozen=True)
class Truth:
    a: Array
    c: Array
    q: Array


def truth(structure: Structure, rng: np.random.Generator) -> Truth:
    m = structure.n_states
    a = np.zeros((m, m))
    q = np.zeros((m, m))
    c = np.zeros((structure.n, m))
    for b in range(structure.n_blocks):
        k = structure.factor_state(b)
        phi = GLOBAL_AR if b == 0 else BLOCK_AR
        a[k, k] = phi
        a[k + 1 : k + LAGS, k : k + LAGS - 1] = np.eye(LAGS - 1)
        q[k, k] = 1.0 - phi**2
    for i in range(structure.n_monthly):
        for blk in (int(v) for v in np.flatnonzero(structure.blocks[i])):
            c[i, structure.factor_state(blk)] = (
                rng.uniform(0.5, 0.9) if blk == 0 else rng.uniform(0.2, 0.5)
            )
        k = structure.monthly_idio_state(i)
        a[k, k] = IDIO_AR
        q[k, k] = IDIO_VARIANCE * (1 - IDIO_AR**2)
        c[i, k] = 1.0
    for j in range(structure.n_quarterly):
        row = structure.n_monthly + j
        for blk in (int(v) for v in np.flatnonzero(structure.blocks[row])):
            k = structure.factor_state(blk)
            load = rng.uniform(0.15, 0.3) if blk == 0 else rng.uniform(0.05, 0.15)
            c[row, k : k + LAGS] = load * TENT
        s = structure.quarterly_idio_states(j)
        a[s.start, s.start] = IDIO_AR
        a[s.start + 1 : s.stop, s.start : s.stop - 1] = np.eye(LAGS - 1)
        q[s.start, s.start] = 0.02
        c[row, s] = TENT
    return Truth(a=a, c=c, q=q)


def draw(
    structure: Structure,
    t: Truth,
    *,
    periods: int,
    rng: np.random.Generator,
    scale: Array | None = None,
    burn: int = 60,
) -> tuple[Array, Array]:
    """(y, z): y (periods, N) with the quarterly series only on third months (rows 2, 5, ...),
    z (periods, m) the states. `scale` multiplies every shock's standard deviation."""
    m = t.a.shape[0]
    sd = np.sqrt(np.diag(t.q))
    s = np.ones(periods) if scale is None else scale
    z = np.zeros(m)
    for _ in range(burn):
        z = t.a @ z + sd * rng.standard_normal(m)
    states = np.zeros((periods, m))
    for k in range(periods):
        z = t.a @ z + s[k] * sd * rng.standard_normal(m)
        states[k] = z
    y = states @ t.c.T + NOISE_SD * rng.standard_normal((periods, structure.n))
    third = (np.arange(periods) % 3) == 2
    y[~third, structure.n_monthly :] = np.nan
    return y, states
