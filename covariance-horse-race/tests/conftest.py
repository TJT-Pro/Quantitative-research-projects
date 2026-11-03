from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from covariance_horse_race.config import ResearchConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> ResearchConfig:
    return load_config(ROOT / "configs" / "research.yaml")


@pytest.fixture(scope="session")
def ci_config() -> ResearchConfig:
    return load_config(ROOT / "configs" / "research.yaml", [ROOT / "configs" / "ci.yaml"])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def factor_covariance(n: int, rng: np.random.Generator, k: int = 3) -> np.ndarray:
    """A daily covariance with a k-factor structure, used as a known truth."""
    loadings = rng.normal(1.0, 0.4, size=(n, k))
    loadings[:, 1:] = rng.normal(0.0, 0.6, size=(n, k - 1))
    factor_var = np.array([1e-4, 3e-5, 2e-5])[:k]
    idio = rng.uniform(2e-5, 2e-4, size=n)
    return (loadings * factor_var) @ loadings.T + np.diag(idio)
