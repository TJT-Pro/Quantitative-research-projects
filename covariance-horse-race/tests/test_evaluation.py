from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize

from covariance_horse_race.config import ResearchConfig
from covariance_horse_race.evaluation import losses
from covariance_horse_race.evaluation.backtest import month_end_positions, run_universe
from covariance_horse_race.evaluation.mcs import model_confidence_set, stationary_bootstrap_indices
from covariance_horse_race.evaluation.portfolio import (
    SingularForecastError,
    drifted_weights,
    gmv_long_only,
    gmv_weights,
)
from covariance_horse_race.evaluation.simulation import multivariate_t, run_simulation
from covariance_horse_race.evaluation.summary import comparisons, describe_status
from tests.conftest import factor_covariance


@settings(max_examples=60, deadline=None)
@given(st.integers(2, 40), st.integers(0, 2**31 - 1), st.floats(0.0, 0.95))
def test_long_only_satisfies_kkt(n: int, seed: int, correlation: float) -> None:
    """Optimality conditions hold for random PD matrices, including strongly
    correlated ones where many weights are pinned at zero."""
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.005, 0.04, n)
    corr = correlation * np.ones((n, n)) + (1 - correlation) * np.eye(n)
    noise = rng.normal(0, 0.05, (n, n))
    sigma = np.outer(vols, vols) * corr + 1e-6 * (noise @ noise.T + np.eye(n))
    w = gmv_long_only(sigma, 1e-12, 1000, 1e-9)
    assert np.all(w >= 0)
    assert w.sum() == pytest.approx(1.0)
    gradient = 2 * sigma @ w
    free = w > 0
    nu = gradient[free].mean()
    assert np.allclose(gradient[free], nu, rtol=1e-6, atol=1e-14)
    assert np.all(gradient[~free] >= nu * (1 - 1e-6))


def test_gmv_weights_minimize_variance(rng: np.random.Generator) -> None:
    sigma = factor_covariance(15, rng)
    w = gmv_weights(sigma, 1e-12)
    assert w.sum() == pytest.approx(1.0)
    best = w @ sigma @ w
    for _ in range(200):
        alt = w + rng.normal(0, 0.05, 15)
        alt -= (alt.sum() - 1) / 15
        assert alt @ sigma @ alt >= best - 1e-18


def test_long_only_matches_slsqp(rng: np.random.Generator) -> None:
    for _ in range(5):
        sigma = factor_covariance(25, rng)
        w = gmv_long_only(sigma, 1e-12, 1000, 1e-9)
        reference = minimize(
            lambda x, s=sigma: x @ s @ x,
            np.full(25, 1 / 25),
            jac=lambda x, s=sigma: 2 * s @ x,
            method="SLSQP",
            bounds=[(0, 1)] * 25,
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
            options={"ftol": 1e-16, "maxiter": 1000},
        )
        assert np.all(w >= 0)
        assert w.sum() == pytest.approx(1.0)
        assert w @ sigma @ w <= reference.fun * (1 + 1e-6)


def test_singular_forecast_detected() -> None:
    x = np.random.default_rng(0).normal(size=(5, 10))
    sigma = x.T @ x
    with pytest.raises(SingularForecastError):
        gmv_weights(sigma, 1e-10)
    with pytest.raises(SingularForecastError):
        gmv_long_only(sigma, 1e-10, 100, 1e-10)


def test_drifted_weights() -> None:
    w = np.array([0.5, 0.5])
    r = np.array([[0.10, 0.0], [0.0, 0.0]])
    assert np.allclose(drifted_weights(w, r), [0.55 / 1.05, 0.5 / 1.05])


def test_qlike_minimized_at_truth(rng: np.random.Generator) -> None:
    sigma = factor_covariance(6, rng)
    proxy_mean = sigma
    assert losses.qlike(proxy_mean, sigma) < losses.qlike(proxy_mean, 1.3 * sigma)
    assert losses.qlike(proxy_mean, sigma) < losses.qlike(proxy_mean, 0.7 * sigma)


def test_qlike_ranking_robust_to_noisy_proxy(rng: np.random.Generator) -> None:
    """With a conditionally unbiased proxy, expected QLIKE still ranks the truth first."""
    sigma = factor_covariance(8, rng)
    wrong = sigma + 2e-5 * np.eye(8)
    gap = []
    for _ in range(400):
        r = rng.multivariate_normal(np.zeros(8), sigma, size=21)
        proxy = losses.realized_proxy(r)
        gap.append(losses.qlike(proxy, wrong) - losses.qlike(proxy, sigma))
    assert np.mean(gap) > 0


def test_stationary_bootstrap_block_structure(rng: np.random.Generator) -> None:
    idx = stationary_bootstrap_indices(500, 200, 10.0, rng)
    assert idx.min() >= 0
    assert idx.max() < 500
    continued = np.mean((idx[:, 1:] - idx[:, :-1]) % 500 == 1)
    assert continued == pytest.approx(0.9, abs=0.02)


def test_mcs_keeps_only_clearly_best_model(rng: np.random.Generator) -> None:
    t = 400
    losses = np.column_stack(
        [rng.normal(1.0, 0.2, t), rng.normal(1.4, 0.2, t), rng.normal(1.8, 0.2, t)]
    )
    result = model_confidence_set(
        losses, ["a", "b", "c"], alpha=0.1, replications=500, mean_block_length=5.0, rng=rng
    )
    assert result.included == ["a"]
    assert result.elimination_order == ["c", "b"]
    assert result.p_values["a"] == 1.0


def test_mcs_size_under_equal_ability(rng: np.random.Generator) -> None:
    """Under equal expected losses the full set survives roughly 1 - alpha of the time."""
    kept = 0
    trials = 60
    for _ in range(trials):
        common = rng.normal(0, 1, 250)
        losses = np.column_stack([common + rng.normal(0, 1, 250) for _ in range(4)])
        result = model_confidence_set(
            losses, list("abcd"), alpha=0.1, replications=300, mean_block_length=3.0, rng=rng
        )
        kept += len(result.included) == 4
    assert kept / trials >= 0.75


def test_mcs_p_values_monotone_in_elimination(rng: np.random.Generator) -> None:
    losses = rng.normal(0, 1, (300, 5)) + np.array([0.0, 0.05, 0.1, 0.3, 0.6])
    result = model_confidence_set(
        losses, list("abcde"), alpha=0.1, replications=400, mean_block_length=4.0, rng=rng
    )
    p = [result.p_values[m] for m in result.elimination_order]
    assert p == sorted(p)


def test_month_end_positions() -> None:
    index = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02"])
    assert list(month_end_positions(index)) == [1, 3, 4]


def _panel(
    rng: np.random.Generator, n_days: int = 900, n: int = 6
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    sigma = factor_covariance(n, rng)
    returns = pd.DataFrame(
        multivariate_t(sigma, n_days, 6.0, rng), index=dates, columns=[f"a{i}" for i in range(n)]
    )
    returns.iloc[:300, 0] = np.nan  # a late-listing asset
    factors = pd.DataFrame(
        rng.normal(0, 0.01, (n_days, 3)), index=dates, columns=["Mkt-RF", "SMB", "HML"]
    )
    extended = factors.assign(RMW=0.0, CMA=0.0, Mom=0.0).iloc[400:]
    return returns, {"primary": factors, "extended": extended}


def _small(config: ResearchConfig) -> ResearchConfig:
    bt = config.backtest.model_copy(update={"windows": [126, 252], "primary_window": 252})
    keep = [m for m in config.models if m.kind != "dcc"]
    return config.model_copy(update={"backtest": bt, "models": keep})


def test_backtest_is_point_in_time(config: ResearchConfig, rng: np.random.Generator) -> None:
    """Changing returns after a rebalance date must not change that date's weights,
    and an asset is admitted only once its window is complete."""
    cfg = _small(config)
    returns, factors = _panel(rng)
    base = run_universe(cfg, "u", returns, factors).monthly
    cutoff = pd.Timestamp(base["rebalance_date"].sort_values().unique()[3])
    shocked = returns.copy()
    shocked.loc[shocked.index > cutoff] *= -7.0
    after = run_universe(cfg, "u", shocked, factors).monthly
    cols = ["n_assets", "status"]
    upto = base["rebalance_date"] < cutoff
    pd.testing.assert_frame_equal(
        base.loc[upto, cols].reset_index(drop=True),
        after.loc[after["rebalance_date"] < cutoff, cols].reset_index(drop=True),
    )
    # Losses of months that end on or before the cutoff are identical as well.
    early = base["rebalance_date"] < pd.Timestamp(base["rebalance_date"].sort_values().unique()[2])
    assert np.allclose(
        base.loc[early, "gmv_variance"].to_numpy(),
        after.loc[
            after["rebalance_date"].isin(base.loc[early, "rebalance_date"]), "gmv_variance"
        ].to_numpy(),
        equal_nan=True,
    )
    # The late asset is excluded until a full window of its returns exists.
    first_full = returns.index[300 + 252 - 1]
    rows = base[(base["window"] == 252) & (base["model"] == "sample")]
    assert (rows.loc[rows["rebalance_date"] < first_full, "n_assets"] == 5).all()
    assert (rows.loc[rows["rebalance_date"] >= first_full, "n_assets"] == 6).all()


def test_backtest_reports_missing_factors(config: ResearchConfig, rng: np.random.Generator) -> None:
    cfg = _small(config)
    returns, factors = _panel(rng)
    monthly = run_universe(cfg, "u", returns, factors).monthly
    ff6 = monthly[monthly["model"] == "ff6"]
    assert (ff6["status"] == "no_factors").any()
    assert (ff6["status"] == "ok").any()
    found = {c.window: c for c in comparisons(monthly, "full")}
    assert "ff6" in found[252].excluded
    assert "ff3" in found[252].models


def test_simulation_shrinkage_beats_sample_in_high_dimension(ci_config: ResearchConfig) -> None:
    sim = ci_config.simulation.model_copy(
        update={"dimensions": [30], "sample_sizes": [60], "replications": 30}
    )
    cfg = ci_config.model_copy(update={"simulation": sim})
    table = run_simulation(cfg).set_index("model")
    assert (
        table.loc["lw_nl", "excess_gmv_variance_pct"]
        < table.loc["sample", "excess_gmv_variance_pct"]
    )
    assert table.loc["lw_nl", "prial"] > 0
    assert table.loc["sample", "prial"] == pytest.approx(0.0, abs=1e-9)


def test_status_codes_read_as_text() -> None:
    assert describe_status("no_factors") == "factor data not yet available"
    assert describe_status("singular") == "forecast covariance is singular"
    assert describe_status("fit_failed: no convergence") == ("estimation failed: no convergence")
    # An unknown code is shown as is rather than hidden.
    assert describe_status("new_code") == "new_code"
