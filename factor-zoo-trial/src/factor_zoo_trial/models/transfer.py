"""Out of sample in space: score the US posterior as a forecast of another country's alphas.

For country c with alpha estimates a_c (sampling covariance S_c) over the factor set F:

    transfer:    a_c ~ N(m_F, V_FF + eta^2 I + S_c)   US posterior plus a country deviation
    strict:      a_c ~ N(m_F, V_FF + S_c)             the same alphas as the US
    dispersion:  a_c ~ N(0, Omega_FF + S_c)           US prior spread, no US direction
    no skill:    a_c ~ N(0, S_c)                      true alphas are zero

eta is estimated by maximum marginal likelihood on every other country, so no country
helps choose the parameter it is scored with. With A = V_FF + S_c = Q L Q', the
transfer density for any eta needs only the eigenvalues: log|A + eta^2 I| = sum log(l + eta^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from factor_zoo_trial.models.empirical_bayes import gaussian_log_density

FloatArray = NDArray[np.float64]
_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class CountryCase:
    name: str
    alpha_hat: FloatArray
    sigma: FloatArray
    us_mean: FloatArray
    us_cov: FloatArray
    us_prior: FloatArray


@dataclass(frozen=True)
class _Spectral:
    eigval: FloatArray
    projected: FloatArray  # Q'(a - m)


def _spectral(case: CountryCase) -> _Spectral:
    a = case.us_cov + case.sigma
    eigval, eigvec = np.linalg.eigh((a + a.T) / 2.0)
    return _Spectral(
        eigval=np.maximum(eigval, 1e-12), projected=eigvec.T @ (case.alpha_hat - case.us_mean)
    )


def _transfer_density(spec: _Spectral, eta: float) -> float:
    var = spec.eigval + eta**2
    return float(
        -0.5 * (np.sum(spec.projected**2 / var) + np.sum(np.log(var)) + var.size * _LOG_2PI)
    )


def estimate_eta(spectra: list[_Spectral]) -> float:
    if not spectra:
        return 0.0
    scale = float(np.sqrt(np.mean([np.mean(s.projected**2) for s in spectra])))
    upper = max(10.0 * scale, 1e-6)

    def objective(eta: float) -> float:
        return -sum(_transfer_density(s, eta) for s in spectra)

    result = optimize.minimize_scalar(objective, bounds=(0.0, upper), method="bounded")
    return float(result.x)


@dataclass(frozen=True)
class CountryScore:
    name: str
    n_factors: int
    eta: float
    transfer: float
    strict: float
    dispersion: float
    no_skill: float

    def per_factor(self, model: str, against: str) -> float:
        gain: float = getattr(self, model) - getattr(self, against)
        return gain / self.n_factors


def score_countries(cases: list[CountryCase]) -> list[CountryScore]:
    spectra = [_spectral(c) for c in cases]
    scores = []
    for i, case in enumerate(cases):
        eta = estimate_eta([s for j, s in enumerate(spectra) if j != i])
        zero = np.zeros(case.alpha_hat.size)
        scores.append(
            CountryScore(
                name=case.name,
                n_factors=int(case.alpha_hat.size),
                eta=eta,
                transfer=_transfer_density(spectra[i], eta),
                strict=_transfer_density(spectra[i], 0.0),
                dispersion=gaussian_log_density(case.alpha_hat, zero, case.us_prior + case.sigma),
                no_skill=gaussian_log_density(case.alpha_hat, zero, case.sigma),
            )
        )
    return scores
