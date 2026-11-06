"""Sampling covariance of factor alphas from an unbalanced panel of returns.

Factors start on different dates, and in most countries there are more factors than
useful months, so the sample correlation matrix is noisy or singular. Correlations are
estimated pairwise and shrunk toward the identity with the Schafer and Strimmer (2005)
intensity (computed from the pairwise overlaps).

A matrix of pairwise correlations from factors with different start dates is not a valid
correlation matrix: on the JKP US panel it has negative eigenvalues even after shrinkage.
Clipping those to a numerical floor would leave directions in which the alpha estimates
look almost exactly known, and every Gaussian score along them explodes. Instead the
eigenvalues are clipped at the Marchenko-Pastur lower edge (1 - sqrt(K / n))^2 for K
factors and the median pairwise overlap n, the smallest eigenvalue a correlation matrix
of independent series estimated from n months can be told apart from (Laloux, Cizeau,
Bouchaud and Potters 1999), and the result is rescaled to a unit diagonal. The alpha
covariance is D R D with D the standard errors of the means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AlphaCovariance:
    sigma: FloatArray  # covariance of the alpha estimates
    correlation: FloatArray  # the shrunk, PSD correlation used
    intensity: float  # shrinkage weight on the identity
    standard_errors: FloatArray
    eigenvalue_floor: float  # lower bound applied to the correlation eigenvalues
    clipped: int  # eigenvalues raised to the floor


def pairwise_correlation(
    returns: FloatArray, *, min_pair: int
) -> tuple[FloatArray, NDArray[np.int64]]:
    values = np.asarray(returns, dtype=np.float64)
    mask = np.isfinite(values).astype(np.float64)
    filled = np.where(mask > 0, values, 0.0)
    overlap = mask.T @ mask
    sx = filled.T @ mask
    sxx = (filled**2).T @ mask
    sxy = filled.T @ filled
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = sx / overlap
        my = sx.T / overlap
        cov = sxy / overlap - mx * my
        vx = sxx / overlap - mx**2
        vy = sxx.T / overlap - my**2
        corr = cov / np.sqrt(vx * vy)
    corr = np.where((overlap >= min_pair) & np.isfinite(corr), corr, 0.0)
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr, overlap.astype(np.int64)


def schafer_strimmer_intensity(corr: FloatArray, overlap: NDArray[np.int64]) -> float:
    """lambda* = sum Var(r_ij) / sum r_ij^2 over i != j, with the normal-theory
    Var(r_ij) = (1 - r_ij^2)^2 / n_ij for each pair's own overlap n_ij."""
    off = ~np.eye(corr.shape[0], dtype=bool) & (overlap > 1) & (corr != 0.0)
    if not off.any():
        return 1.0
    r = corr[off]
    variance = (1.0 - r**2) ** 2 / overlap[off]
    denominator = float(np.sum(r**2))
    if denominator <= 0:
        return 1.0
    return float(np.clip(np.sum(variance) / denominator, 0.0, 1.0))


def marchenko_pastur_floor(*, n_factors: int, n_months: float, minimum: float) -> float:
    """(1 - sqrt(K / n))^2, never below `minimum` (the edge is zero when K equals n)."""
    if n_months <= 0:
        return 1.0
    edge = (1.0 - np.sqrt(n_factors / n_months)) ** 2
    return float(min(1.0, max(edge, minimum)))


def median_overlap(overlap: NDArray[np.int64], *, min_pair: int) -> float:
    """Median months shared by a pair of factors, over the pairs that enter the matrix."""
    off = ~np.eye(overlap.shape[0], dtype=bool) & (overlap >= min_pair)
    if not off.any():
        return float(np.median(np.diag(overlap)))
    return float(np.median(overlap[off]))


def nearest_correlation(corr: FloatArray, *, floor: float = 1e-8) -> FloatArray:
    """Raise eigenvalues below `floor` to it and rescale back to a unit diagonal."""
    sym = (corr + corr.T) / 2.0
    eigval, eigvec = np.linalg.eigh(sym)
    clipped = (eigvec * np.maximum(eigval, floor)) @ eigvec.T
    scale = 1.0 / np.sqrt(np.diag(clipped))
    out = clipped * scale[:, None] * scale[None, :]
    np.fill_diagonal(out, 1.0)
    return np.asarray(out, dtype=np.float64)


def alpha_covariance(
    returns: FloatArray,
    *,
    min_pair: int,
    shrinkage: Literal["schafer_strimmer", "none"],
    eigenvalue_floor: Literal["marchenko_pastur", "numerical"],
    minimum_floor: float,
) -> AlphaCovariance:
    values = np.asarray(returns, dtype=np.float64)
    counts = np.isfinite(values).sum(axis=0)
    sd = np.nanstd(values, axis=0, ddof=1)
    se = sd / np.sqrt(counts)
    corr, overlap = pairwise_correlation(values, min_pair=min_pair)
    intensity = schafer_strimmer_intensity(corr, overlap) if shrinkage != "none" else 0.0
    shrunk = (1.0 - intensity) * corr + intensity * np.eye(corr.shape[0])
    if eigenvalue_floor == "marchenko_pastur":
        floor = marchenko_pastur_floor(
            n_factors=corr.shape[0],
            n_months=median_overlap(overlap, min_pair=min_pair),
            minimum=minimum_floor,
        )
    else:
        floor = 1e-8
    clipped = int(np.sum(np.linalg.eigvalsh((shrunk + shrunk.T) / 2.0) < floor))
    psd = nearest_correlation(shrunk, floor=floor)
    sigma = psd * se[:, None] * se[None, :]
    return AlphaCovariance(
        sigma=sigma,
        correlation=psd,
        intensity=intensity,
        standard_errors=se,
        eigenvalue_floor=floor,
        clipped=clipped,
    )
