"""Model Confidence Set (Hansen, Lunde and Nason, 2011, Econometrica 79(2)).

Uses the T_max statistic with the elimination rule e = argmax_i t_i, where

    t_i = dbar_i / sqrt(var(dbar_i)),   dbar_i = mean_t(L_it - mean_j L_jt) over the
    surviving set,

and the stationary bootstrap (Politis and Romano, 1994) for both the variance and
the null distribution. Bootstrap means of the raw losses are drawn once; every
statistic for every candidate set is a linear function of them, so the whole
elimination sequence costs one pass of resampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def stationary_bootstrap_indices(
    n: int, replications: int, mean_block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Index matrix (replications x n) for the stationary bootstrap."""
    if n < 2:
        raise ValueError("need at least two observations")
    p = 1.0 / mean_block_length
    indices = np.empty((replications, n), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n, size=replications)
    new_block = rng.random((replications, n)) < p
    fresh = rng.integers(0, n, size=(replications, n))
    for t in range(1, n):
        continued = (indices[:, t - 1] + 1) % n
        indices[:, t] = np.where(new_block[:, t], fresh[:, t], continued)
    return indices


@dataclass(frozen=True)
class McsResult:
    models: list[str]
    p_values: dict[str, float]
    included: list[str]
    elimination_order: list[str]
    alpha: float

    def contains(self, model: str) -> bool:
        return model in self.included


def model_confidence_set(
    losses: np.ndarray,
    names: list[str],
    *,
    alpha: float,
    replications: int,
    mean_block_length: float,
    rng: np.random.Generator,
) -> McsResult:
    """Run the MCS on a T x M loss matrix with no missing values."""
    if losses.ndim != 2 or losses.shape[1] != len(names):
        raise ValueError("losses must be T x M matching names")
    if not np.isfinite(losses).all():
        raise ValueError("losses contain non-finite values")
    t, m = losses.shape
    sample_means = losses.mean(axis=0)
    idx = stationary_bootstrap_indices(t, replications, mean_block_length, rng)
    boot_means = losses[idx].mean(axis=1)  # B x M
    centred = boot_means - sample_means

    # The elimination runs to a single model; each model's MCS p-value is the
    # running maximum of the equivalence-test p-values up to its elimination.
    # The set at level alpha is then {i : p_i >= alpha} (HLN, Definition 2).
    surviving = list(range(m))
    eliminated: list[int] = []
    p_values = np.ones(m)
    running_max = 0.0
    while len(surviving) > 1:
        s = np.array(surviving)
        dbar = sample_means[s] - sample_means[s].mean()
        dstar = centred[:, s] - centred[:, s].mean(axis=1, keepdims=True)
        var = np.mean(dstar**2, axis=0)
        if np.any(var <= 0):
            # Losses identical across the remaining models: they are equivalent.
            break
        tstat = dbar / np.sqrt(var)
        tmax = float(tstat.max())
        tmax_star = np.max(dstar / np.sqrt(var), axis=1)
        p = float(np.mean(tmax_star >= tmax))
        running_max = max(running_max, p)
        worst = int(s[int(np.argmax(tstat))])
        p_values[worst] = running_max
        eliminated.append(worst)
        surviving.remove(worst)
    for index in surviving:
        p_values[index] = 1.0
    included = [names[i] for i in range(m) if p_values[i] >= alpha]
    return McsResult(
        models=list(names),
        p_values={names[i]: float(p_values[i]) for i in range(m)},
        included=included,
        elimination_order=[names[i] for i in eliminated],
        alpha=alpha,
    )
