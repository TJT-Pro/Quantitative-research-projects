from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from acm_term_premium.config import ResearchConfig, load_config
from acm_term_premium.models import acm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> ResearchConfig:
    return load_config(ROOT / "configs" / "research.yaml")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2026)


def simulate_affine(
    rng: np.random.Generator, months: int = 600, noise_bp: float = 0.5
) -> tuple[pd.DataFrame, acm.Parameters, np.ndarray]:
    """Yields from a known three-factor affine model plus small measurement error.

    Returns the curve (1..120 months, annual decimals), the true parameters and
    the true factor path, so an estimator can be checked against the truth.
    """
    k = 3
    # Moderate persistence keeps the VAR's small-sample bias negligible, so the
    # estimator can be held to the truth rather than to a biased target.
    phi = np.diag([0.95, 0.90, 0.80])
    sigma = np.diag([0.30, 0.25, 0.20]) ** 2
    mu = np.zeros(k)
    delta0, delta1 = 0.004, np.array([0.0012, 0.0006, 0.0003])
    lambda1 = np.diag([-0.02, 0.04, 0.02])
    lambda0 = np.array([-0.05, 0.02, 0.0])
    truth = acm.Parameters(mu, phi, sigma, 1e-8, delta0, delta1, lambda0, lambda1, 0.95, False)
    a, b = acm._recursions(truth, lambda0, lambda1, 120, "paper")
    chol = np.linalg.cholesky(sigma)
    x = np.zeros((months, k))
    for t in range(1, months):
        x[t] = phi @ x[t - 1] + chol @ rng.standard_normal(k)
    n = np.arange(1, 121)
    yields = -(a[n] + x @ b[n].T) / n * 12.0
    yields += rng.normal(0.0, noise_bp * 1e-4, yields.shape)
    dates = pd.date_range("1970-01-31", periods=months, freq="ME")
    return pd.DataFrame(yields, index=dates, columns=list(n)), truth, x
