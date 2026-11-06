"""Multiple-testing procedures on one-sided tests of H0: mean <= 0.

Bonferroni and Holm control the family-wise error rate under any dependence;
Benjamini-Hochberg controls the false discovery rate under independence or positive
dependence; Benjamini-Yekutieli under arbitrary dependence; Harvey-Liu-Zhu is a
fixed t hurdle; Romano-Wolf controls the family-wise error rate with a bootstrap that
keeps the cross-sectional dependence; Storey estimates the share of true nulls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _check(p: FloatArray) -> FloatArray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or np.any((p < 0) | (p > 1)) or np.any(~np.isfinite(p)):
        raise ValueError("p-values must be a non-empty vector in [0, 1]")
    return p


def unadjusted(p: FloatArray, *, level: float) -> BoolArray:
    return _check(p) <= level


def bonferroni(p: FloatArray, *, level: float) -> BoolArray:
    p = _check(p)
    return p <= level / p.size


def holm(p: FloatArray, *, level: float) -> BoolArray:
    p = _check(p)
    m = p.size
    order = np.argsort(p, kind="stable")
    thresholds = level / (m - np.arange(m))
    passed = p[order] <= thresholds
    stop = m if passed.all() else int(np.argmin(passed))
    reject = np.zeros(m, dtype=bool)
    reject[order[:stop]] = True
    return reject


def _step_up(p: FloatArray, *, level: float) -> BoolArray:
    m = p.size
    order = np.argsort(p, kind="stable")
    below = p[order] <= level * np.arange(1, m + 1) / m
    reject = np.zeros(m, dtype=bool)
    if below.any():
        last = int(np.nonzero(below)[0].max())
        reject[order[: last + 1]] = True
    return reject


def benjamini_hochberg(p: FloatArray, *, level: float) -> BoolArray:
    return _step_up(_check(p), level=level)


def benjamini_yekutieli(p: FloatArray, *, level: float) -> BoolArray:
    p = _check(p)
    harmonic = float(np.sum(1.0 / np.arange(1, p.size + 1)))
    return _step_up(p, level=level / harmonic)


def harvey_liu_zhu(t: FloatArray, *, threshold: float) -> BoolArray:
    return np.asarray(t, dtype=np.float64) > threshold


@dataclass(frozen=True)
class StoreyEstimate:
    pi0: float  # at the fixed lambda
    fixed_lambda: float
    pi0_bootstrap: float  # at the bootstrap-chosen lambda
    chosen_lambda: float
    by_lambda: FloatArray


def storey_fixed(p: FloatArray, *, lam: float) -> float:
    """Storey (2002): #{p > lambda} / (m (1 - lambda)), capped at one. Conservative when
    null p-values are uniform, because non-null p-values above lambda only add to it."""
    p = _check(p)
    return float(min((p > lam).sum() / (p.size * (1.0 - lam)), 1.0))


def storey_pi0(
    p: FloatArray,
    *,
    lam: float,
    lambdas: tuple[float, ...],
    draws: int,
    rng: np.random.Generator,
) -> StoreyEstimate:
    """Share of true nulls at a fixed lambda, plus the bootstrap MSE choice of Storey,
    Taylor and Siegmund (2004) for comparison. The bootstrap rule targets the smallest
    full-sample estimate, which biases it downward; the simulation study measures by how much."""
    p = _check(p)
    grid = np.asarray(lambdas, dtype=np.float64)
    m = p.size

    def estimate(sample: FloatArray) -> FloatArray:
        above = (sample[None, :] > grid[:, None]).sum(axis=1)
        return np.minimum(above / (m * (1.0 - grid)), 1.0)

    full = estimate(p)
    target = float(full.min())
    boot = np.empty((draws, grid.size))
    for b in range(draws):
        boot[b] = estimate(p[rng.integers(0, m, size=m)])
    mse = ((boot - target) ** 2).mean(axis=0)
    best = int(np.argmin(mse))
    return StoreyEstimate(
        pi0=storey_fixed(p, lam=lam),
        fixed_lambda=lam,
        pi0_bootstrap=float(full[best]),
        chosen_lambda=float(grid[best]),
        by_lambda=full,
    )


# Romano-Wolf ------------------------------------------------------------------------------


@dataclass(frozen=True)
class RomanoWolf:
    t: FloatArray  # studentized statistic per hypothesis
    adjusted_p: FloatArray
    reject: BoolArray


def circular_block_counts(
    *, n_months: int, block: int, draws: int, rng: np.random.Generator
) -> FloatArray:
    """How often each month appears in each circular-block bootstrap sample."""
    n_blocks = -(-n_months // block)
    counts = np.zeros((draws, n_months))
    offsets = np.arange(block)
    for b in range(draws):
        starts = rng.integers(0, n_months, size=n_blocks)
        picks = ((starts[:, None] + offsets[None, :]) % n_months).ravel()[:n_months]
        counts[b] = np.bincount(picks, minlength=n_months)
    return counts


def _moments(
    weights: FloatArray, values: FloatArray, mask: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    n = weights @ mask
    s1 = weights @ values
    s2 = weights @ (values**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = (s2 - n * mean**2) / (n - 1)
        se = np.sqrt(np.maximum(var, 0.0) / n)
    return mean, se, n


def romano_wolf(
    returns: FloatArray,
    *,
    level: float,
    draws: int,
    block: int,
    rng: np.random.Generator,
) -> RomanoWolf:
    """Studentized step-down on means of the columns of `returns` (months x hypotheses,
    NaN where a hypothesis has no observation). Each bootstrap sample draws whole months,
    so every hypothesis sees the same resampled calendar and the dependence across
    hypotheses is preserved."""
    values = np.asarray(returns, dtype=np.float64)
    mask = np.isfinite(values).astype(np.float64)
    filled = np.where(mask > 0, values, 0.0)
    n_months, m = values.shape
    ones = np.ones((1, n_months))
    mean, se, _ = _moments(ones, filled, mask)
    t_obs = (mean / se).ravel()
    counts = circular_block_counts(n_months=n_months, block=block, draws=draws, rng=rng)
    b_mean, b_se, b_n = _moments(counts, filled, mask)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_boot = (b_mean - mean) / b_se
    t_boot = np.where((b_n >= 2) & np.isfinite(t_boot), t_boot, -np.inf)

    order = np.argsort(-t_obs, kind="stable")
    sorted_boot = t_boot[:, order]
    # Max over the hypotheses still in play at each step: suffix maxima in sorted order.
    suffix_max = np.maximum.accumulate(sorted_boot[:, ::-1], axis=1)[:, ::-1]
    raw = (suffix_max >= t_obs[order][None, :]).mean(axis=0)
    adjusted_sorted = np.maximum.accumulate(raw)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return RomanoWolf(t=t_obs, adjusted_p=adjusted, reject=adjusted <= level)
