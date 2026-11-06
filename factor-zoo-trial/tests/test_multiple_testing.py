"""Properties every multiple-testing procedure must satisfy, and Romano-Wolf behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from factor_zoo_trial.models import multiple_testing as mt

p_values = arrays(
    np.float64,
    st.integers(min_value=1, max_value=300),
    elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
levels = st.floats(min_value=0.001, max_value=0.5)


@given(p=p_values, level=levels)
def test_adjusted_never_reject_more_than_unadjusted(p: np.ndarray, level: float) -> None:
    raw = mt.unadjusted(p, level=level)
    for procedure in (mt.bonferroni, mt.holm, mt.benjamini_hochberg, mt.benjamini_yekutieli):
        adjusted = procedure(p, level=level)
        assert not np.any(adjusted & ~raw)


@given(p=p_values, level=levels)
def test_holm_rejects_at_least_bonferroni(p: np.ndarray, level: float) -> None:
    bonf = mt.bonferroni(p, level=level)
    holm = mt.holm(p, level=level)
    assert np.all(holm[bonf])


@given(p=p_values, level=levels)
def test_bhy_is_within_bh_and_bh_within_holm_complement(p: np.ndarray, level: float) -> None:
    bh = mt.benjamini_hochberg(p, level=level)
    bhy = mt.benjamini_yekutieli(p, level=level)
    holm = mt.holm(p, level=level)
    assert not np.any(bhy & ~bh)
    assert not np.any(holm & ~bh)  # step-up FDR control is never stricter than Holm


@given(p=p_values, level=levels)
def test_rejections_are_monotone_in_p(p: np.ndarray, level: float) -> None:
    for procedure in (mt.bonferroni, mt.holm, mt.benjamini_hochberg, mt.benjamini_yekutieli):
        reject = procedure(p, level=level)
        if reject.any() and (~reject).any():
            assert p[reject].max() <= p[~reject].min()


@given(p=p_values, lam=st.floats(min_value=0.05, max_value=0.95))
def test_storey_fixed_in_unit_interval(p: np.ndarray, lam: float) -> None:
    assert 0.0 <= mt.storey_fixed(p, lam=lam) <= 1.0


def test_storey_is_near_truth_for_uniform_nulls() -> None:
    rng = np.random.default_rng(3)
    p = np.concatenate([rng.uniform(size=8000), rng.uniform(0, 1e-4, size=2000)])
    est = mt.storey_pi0(p, lam=0.5, lambdas=(0.2, 0.5, 0.8), draws=50, rng=rng)
    assert est.pi0 == pytest.approx(0.8, abs=0.03)


def test_rejects_bad_p_values() -> None:
    with pytest.raises(ValueError, match="p-values"):
        mt.holm(np.array([0.1, 1.2]), level=0.05)


def test_circular_block_counts_sum_to_months() -> None:
    counts = mt.circular_block_counts(
        n_months=100, block=12, draws=20, rng=np.random.default_rng(0)
    )
    assert np.all(counts.sum(axis=1) == 100)


@settings(deadline=None, max_examples=10)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_romano_wolf_adjusted_p_monotone_in_t(seed: int) -> None:
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((120, 15)) + np.linspace(0, 0.4, 15)
    rw = mt.romano_wolf(returns, level=0.05, draws=200, block=1, rng=rng)
    order = np.argsort(-rw.t)
    assert np.all(np.diff(rw.adjusted_p[order]) >= -1e-12)


def test_romano_wolf_controls_fwer_under_the_global_null() -> None:
    rng = np.random.default_rng(11)
    common = rng.standard_normal((200, 1))
    false = 0
    trials = 60
    for _ in range(trials):
        x = 0.7 * common @ np.ones((1, 30)) + rng.standard_normal((200, 30))
        common = rng.standard_normal((200, 1))
        false += int(mt.romano_wolf(x, level=0.1, draws=300, block=1, rng=rng).reject.any())
    assert false / trials <= 0.1 + 2 * np.sqrt(0.1 * 0.9 / trials)


def test_romano_wolf_finds_strong_effects_with_missing_months() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((300, 10))
    x[:, :3] += 1.0
    x[:100, 5] = np.nan  # a hypothesis whose window starts later
    rw = mt.romano_wolf(x, level=0.05, draws=300, block=6, rng=rng)
    assert rw.reject[:3].all()
    assert not rw.reject[3:].any()
