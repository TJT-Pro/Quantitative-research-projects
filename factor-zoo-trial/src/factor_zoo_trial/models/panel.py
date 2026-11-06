"""Panel regression with group fixed effects and standard errors clustered by month.

    y_it = a_i + x_it' b + e_it

The fixed effects are kept as explicit coefficients (not demeaned away) because the
headline needs the covariance between the average fixed effect and the slopes. The
design is never built densely: D'D and the per-cluster scores are accumulated by group
sums, so a 200-predictor, 1,200-month panel needs a (200 + k)-square system only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class FixedEffectsFit:
    effects: FloatArray  # one per group
    slopes: FloatArray  # one per regressor
    covariance: FloatArray  # joint, (groups + regressors) square, cluster-robust
    n_obs: int
    n_groups: int
    n_clusters: int

    def linear(self, weights: FloatArray) -> tuple[float, float]:
        """Estimate and standard error of weights' [effects, slopes]."""
        theta = np.concatenate([self.effects, self.slopes])
        value = float(weights @ theta)
        variance = float(weights @ self.covariance @ weights)
        return value, float(np.sqrt(max(variance, 0.0)))


def fit_fixed_effects(
    *, y: FloatArray, groups: IntArray, clusters: IntArray, regressors: FloatArray
) -> FixedEffectsFit:
    y = np.asarray(y, dtype=np.float64)
    x = np.atleast_2d(np.asarray(regressors, dtype=np.float64))
    if x.shape[0] != y.size:
        x = x.T
    n, k = x.shape
    n_groups = int(groups.max()) + 1
    _, cluster_codes = np.unique(clusters, return_inverse=True)
    n_clusters = int(cluster_codes.max()) + 1
    p = n_groups + k

    counts = np.bincount(groups, minlength=n_groups).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("every group needs at least one observation")
    cross = np.zeros((n_groups, k))
    np.add.at(cross, groups, x)
    gram = np.zeros((p, p))
    gram[:n_groups, :n_groups] = np.diag(counts)
    gram[:n_groups, n_groups:] = cross
    gram[n_groups:, :n_groups] = cross.T
    gram[n_groups:, n_groups:] = x.T @ x

    moment = np.zeros(p)
    moment[:n_groups] = np.bincount(groups, weights=y, minlength=n_groups)
    moment[n_groups:] = x.T @ y
    bread = np.linalg.inv(gram)
    theta = bread @ moment
    resid = y - theta[groups] - x @ theta[n_groups:]

    scores = np.zeros((n_clusters, p))
    np.add.at(scores, (cluster_codes, groups), resid)
    xe = x * resid[:, None]
    np.add.at(scores[:, n_groups:], cluster_codes, xe)
    meat = scores.T @ scores
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - p))
    covariance = correction * bread @ meat @ bread
    return FixedEffectsFit(
        effects=theta[:n_groups],
        slopes=theta[n_groups:],
        covariance=covariance,
        n_obs=n,
        n_groups=n_groups,
        n_clusters=n_clusters,
    )
