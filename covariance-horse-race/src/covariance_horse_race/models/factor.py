"""Factor-structured covariance: observable Fama-French factors and PCA factors."""

from __future__ import annotations

import numpy as np

from covariance_horse_race.models.base import (
    CovarianceModel,
    Matrix,
    check_window,
    demean,
    symmetrize,
)


def observable_factor_covariance(returns: Matrix, factors: Matrix) -> Matrix:
    """B Sigma_f B' + D from time-series OLS of each asset on the factors.

    Residual variances use T - K - 1 degrees of freedom. D is diagonal, which is
    the restriction that makes the model a shrinkage of the sample covariance.
    """
    check_window(returns)
    check_window(factors)
    t, k = factors.shape
    if returns.shape[0] != t:
        raise ValueError("returns and factors must cover the same dates")
    if t <= k + 1:
        raise ValueError("window shorter than the number of regressors")
    design = np.column_stack([np.ones(t), factors])
    coef, *_ = np.linalg.lstsq(design, returns, rcond=None)
    betas = coef[1:].T  # N x K
    residuals = returns - design @ coef
    resid_var = (residuals**2).sum(axis=0) / (t - k - 1)
    f = demean(factors)
    sigma_f = f.T @ f / (t - 1)
    sigma = betas @ sigma_f @ betas.T + np.diag(resid_var)
    return symmetrize(sigma)


class ObservableFactor(CovarianceModel):
    requires_factors = True

    def __init__(self) -> None:
        self._sigma: Matrix | None = None

    def fit(self, returns: Matrix, factors: Matrix | None = None) -> None:
        if factors is None:
            raise ValueError("observable factor model needs factor returns")
        self._sigma = observable_factor_covariance(returns, factors)

    def forecast(self, horizon: int) -> Matrix:
        if self._sigma is None:
            raise RuntimeError("fit before forecast")
        return self._sigma.copy()


def bai_ng_ic2(returns: Matrix, kmax: int) -> int:
    """Number of factors minimizing Bai and Ng (2002) IC_p2 on standardized data."""
    x = demean(returns)
    t, n = x.shape
    sd = x.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("an asset has zero variance in the window")
    z = x / sd
    eigenvalues = np.sort(np.linalg.eigvalsh(z.T @ z / t))[::-1]
    upper = int(min(kmax, n - 1, t - 1))
    penalty = (n + t) / (n * t) * np.log(min(n, t))
    best_k, best_ic = 0, np.inf
    for k in range(upper + 1):
        v = eigenvalues[k:].sum() / n
        if v <= 0:
            break
        ic = np.log(v) + k * penalty
        if ic < best_ic:
            best_k, best_ic = k, ic
    return best_k


def pca_factor_covariance(returns: Matrix, k: int) -> Matrix:
    """Top-k principal components of the covariance plus the diagonal remainder."""
    check_window(returns)
    x = demean(returns)
    t = x.shape[0]
    sample = symmetrize(x.T @ x / (t - 1))
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    systematic = (eigenvectors[:, :k] * eigenvalues[:k]) @ eigenvectors[:, :k].T
    residual = np.clip(np.diag(sample - systematic), a_min=0.0, a_max=None)
    floor = 1e-12 * float(np.mean(np.diag(sample)))
    return symmetrize(systematic + np.diag(np.maximum(residual, floor)))


class StatisticalFactor(CovarianceModel):
    def __init__(self, kmax: int) -> None:
        if kmax < 1:
            raise ValueError("kmax must be positive")
        self.kmax = kmax
        self.k: int | None = None
        self._sigma: Matrix | None = None

    def fit(self, returns: Matrix, factors: Matrix | None = None) -> None:
        self.k = bai_ng_ic2(returns, self.kmax)
        self._sigma = pca_factor_covariance(returns, self.k)

    def forecast(self, horizon: int) -> Matrix:
        if self._sigma is None:
            raise RuntimeError("fit before forecast")
        return self._sigma.copy()
