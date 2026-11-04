"""The estimator called with the settings of a research config, in one place."""

from __future__ import annotations

import pandas as pd

from acm_term_premium.config import ModelConfig
from acm_term_premium.models.acm import AcmModel, Convention, fit


def fit_model(
    curve: pd.DataFrame,
    settings: ModelConfig,
    *,
    convention: Convention | None = None,
    bounded: bool = True,
) -> AcmModel:
    """Fit with `settings`; `convention` overrides the configured one, `bounded=False`
    drops the spectral-radius bound to recover the unconstrained OLS estimate."""
    return fit(
        curve,
        factor_maturities=settings.pc_maturities.months(),
        n_factors=settings.n_factors,
        return_maturities=settings.return_maturities,
        max_maturity=settings.max_maturity,
        convention=convention or settings.convention,
        max_spectral_radius=settings.max_spectral_radius if bounded else None,
    )
