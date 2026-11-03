"""Linear and nonlinear shrinkage of the sample covariance matrix.

Linear: Ledoit and Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix",
shrinking toward the constant-correlation target with the estimated optimal
intensity.

Nonlinear: Ledoit and Wolf (2020), "Analytical Nonlinear Shrinkage of
Large-Dimensional Covariance Matrices", Annals of Statistics 48(5). Keeps the
sample eigenvectors and replaces each eigenvalue with a kernel estimate of its
oracle value, using the Epanechnikov kernel and its Hilbert transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from covariance_horse_race.models.base import (
    CovarianceModel,
    Matrix,
    check_window,
    demean,
    symmetrize,
)

_SQRT5 = float(np.sqrt(5.0))


@dataclass(frozen=True)
class LinearShrinkageResult:
    sigma: Matrix
    intensity: float
    target: Matrix


def constant_correlation_shrinkage(returns: Matrix) -> LinearShrinkageResult:
    check_window(returns)
    x = demean(returns)
    t, n = x.shape
    if n < 2:
        raise ValueError("constant-correlation shrinkage needs at least two assets")
    sample = x.T @ x / t
    var = np.diag(sample).copy()
    if np.any(var <= 0):
        raise ValueError("an asset has zero variance in the window")
    sd = np.sqrt(var)
    corr = sample / np.outer(sd, sd)
    rbar = (corr.sum() - n) / (n * (n - 1))
    target = rbar * np.outer(sd, sd)
    np.fill_diagonal(target, var)

    y = x**2
    phi_mat = y.T @ y / t - 2.0 * (x.T @ x) * sample / t + sample**2
    phi = phi_mat.sum()

    term1 = (x**3).T @ x / t
    term2 = var[:, None] * sample
    term3 = (x.T @ x / t) * var[:, None]
    term4 = var[:, None] * sample
    theta = term1 - term2 - term3 + term4
    np.fill_diagonal(theta, 0.0)
    rho = np.trace(phi_mat) + rbar * (np.outer(1.0 / sd, sd) * theta).sum()

    gamma = float(np.linalg.norm(sample - target, "fro") ** 2)
    if gamma == 0.0:
        intensity = 1.0
    else:
        kappa = (phi - rho) / gamma
        intensity = float(max(0.0, min(1.0, kappa / t)))
    sigma = intensity * target + (1.0 - intensity) * sample
    return LinearShrinkageResult(sigma=symmetrize(sigma), intensity=intensity, target=target)


class LinearShrinkage(CovarianceModel):
    def __init__(self) -> None:
        self._sigma: Matrix | None = None
        self.intensity: float | None = None

    def fit(self, returns: Matrix, factors: Matrix | None = None) -> None:
        result = constant_correlation_shrinkage(returns)
        self._sigma, self.intensity = result.sigma, result.intensity

    def forecast(self, horizon: int) -> Matrix:
        if self._sigma is None:
            raise RuntimeError("fit before forecast")
        return self._sigma.copy()


def analytical_nonlinear_shrinkage(returns: Matrix, demeaned: bool = False) -> Matrix:
    """Ledoit-Wolf (2020) analytical nonlinear shrinkage.

    `demeaned=True` means the caller already removed means estimated elsewhere,
    so no degree of freedom is lost; otherwise the window is demeaned here and
    the effective sample size is T - 1.
    """
    check_window(returns)
    if demeaned:
        x = returns
        n_eff = returns.shape[0]
    else:
        x = demean(returns)
        n_eff = returns.shape[0] - 1
    p = x.shape[1]
    n = n_eff
    sample = symmetrize(x.T @ x / n)
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    lam = eigenvalues[max(0, p - n) :]
    if np.any(lam <= 0):
        raise ValueError("sample covariance has non-positive eigenvalues in its column space")
    h = n ** (-1.0 / 3.0)
    if _SQRT5 * h >= 1.0:
        raise ValueError("window too short for the analytical bandwidth (need T - 1 > 11)")

    L = np.tile(lam, (lam.size, 1)).T  # L[i, j] = lambda_i
    H = h * L.T  # H[i, j] = h * lambda_j
    xm = (L - L.T) / H
    ftilde = (3.0 / 4.0 / _SQRT5) * np.mean(np.maximum(1.0 - xm**2 / 5.0, 0.0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log(np.abs((_SQRT5 - xm) / (_SQRT5 + xm)))
    hf_temp = (-3.0 / 10.0 / np.pi) * xm + (3.0 / 4.0 / _SQRT5 / np.pi) * (
        1.0 - xm**2 / 5.0
    ) * log_term
    edge = np.isclose(np.abs(xm), _SQRT5, rtol=0.0, atol=1e-14)
    hf_temp[edge] = (-3.0 / 10.0 / np.pi) * xm[edge]
    hftilde = np.mean(hf_temp / H, axis=1)

    if p <= n:
        c = p / n
        dtilde = lam / (
            (np.pi * c * lam * ftilde) ** 2 + (1.0 - c - np.pi * c * lam * hftilde) ** 2
        )
    else:
        hf0 = (
            (1.0 / np.pi)
            * (
                3.0 / 10.0 / h**2
                + 3.0
                / 4.0
                / _SQRT5
                / h
                * (1.0 - 1.0 / 5.0 / h**2)
                * np.log((1.0 + _SQRT5 * h) / (1.0 - _SQRT5 * h))
            )
            * np.mean(1.0 / lam)
        )
        d0 = 1.0 / (np.pi * (p - n) / n * hf0)
        d1 = lam / (np.pi**2 * lam**2 * (ftilde**2 + hftilde**2))
        dtilde = np.concatenate([np.full(p - n, d0), d1])
    out: Matrix = (eigenvectors * dtilde) @ eigenvectors.T
    return symmetrize(out)


class NonlinearShrinkage(CovarianceModel):
    def __init__(self) -> None:
        self._sigma: Matrix | None = None

    def fit(self, returns: Matrix, factors: Matrix | None = None) -> None:
        self._sigma = analytical_nonlinear_shrinkage(returns)

    def forecast(self, horizon: int) -> Matrix:
        if self._sigma is None:
            raise RuntimeError("fit before forecast")
        return self._sigma.copy()
