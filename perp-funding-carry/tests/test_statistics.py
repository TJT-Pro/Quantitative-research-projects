"""Forecast alignment and timing, Sharpe and deflated Sharpe, the counterparty split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from perp_funding_carry.models import counterparty, forecast, stats


def _funding(n: int, *, phi: float, seed: int, symbols: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2023-01-01 08:00", periods=n, freq="8h", tz="UTC")
    rows = []
    for s in range(symbols):
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.0001 + phi * (x[i - 1] - 0.0001) + rng.normal(0, 0.0001)
        rows.append(pd.DataFrame({"symbol": f"S{s}", "time": times, "interval_h": 8, "rate": x}))
    return pd.concat(rows, ignore_index=True)


def _bars(funding: pd.DataFrame) -> pd.DataFrame:
    out = []
    for symbol, g in funding.groupby("symbol"):
        opens = pd.date_range(
            g["time"].min() - pd.Timedelta(hours=8), g["time"].max(), freq="4h", tz="UTC"
        )
        out.append(
            pd.DataFrame(
                {"symbol": symbol, "open_time": opens, "mark_close": 100.0, "spot_close": 100.0}
            )
        )
    return pd.concat(out, ignore_index=True)


def test_window_funding_sums_shorter_intervals() -> None:
    times = pd.to_datetime(["2024-01-01 04:00", "2024-01-01 08:00", "2024-01-01 12:00"], utc=True)
    f = pd.DataFrame({"symbol": "X", "time": times, "rate": [0.1, 0.2, 0.3]})
    w = forecast.window_funding(f)
    assert w["f"].tolist() == pytest.approx([0.3, 0.3])


def test_targets_are_the_next_windows_and_features_the_past() -> None:
    f = _funding(80, phi=0.5, seed=1, symbols=1)
    panel = forecast.build_panel(
        f, _bars(f), bar_hours=4, har_lags=(1, 3), momentum_hours=(24,), horizons=(1, 3)
    )
    panel = panel.set_index("t")
    t = panel.index[30]
    window = panel["f"]
    assert panel.loc[t, "y1"] == pytest.approx(window.iloc[31])
    assert panel.loc[t, "y3"] == pytest.approx(window.iloc[31:34].mean())
    assert panel.loc[t, "har3"] == pytest.approx(window.iloc[28:31].mean())


def test_walk_forward_trains_only_on_settled_targets() -> None:
    f = _funding(900, phi=0.9, seed=2)
    panel = forecast.build_panel(
        f, _bars(f), bar_hours=4, har_lags=(1, 3, 21), momentum_hours=(24,), horizons=(1, 21)
    )
    models = forecast.model_features(har_lags=(1, 3, 21), momentum_hours=(24,))
    base = forecast.walk_forward(
        panel, horizons=(21,), models=models, min_train_months=3, winsor=0.0
    )
    # Changing targets that settle after a month starts must not change that month's forecasts.
    month = base["t"].dt.tz_localize(None).dt.to_period("M").min()
    start = pd.Timestamp(month.start_time, tz="UTC")
    shaken = panel.copy()
    late = shaken["t"] + pd.Timedelta(hours=8) * 21 > start
    shaken.loc[late, "y21"] = 99.0
    moved = forecast.walk_forward(
        shaken, horizons=(21,), models=models, min_train_months=3, winsor=0.0
    )
    first = base["t"].dt.tz_localize(None).dt.to_period("M") == month
    np.testing.assert_allclose(base.loc[first, "har"], moved.loc[first, "har"])


def test_har_beats_persistence_when_funding_mean_reverts() -> None:
    f = _funding(1500, phi=0.6, seed=3)
    panel = forecast.build_panel(
        f, _bars(f), bar_hours=4, har_lags=(1, 3, 21), momentum_hours=(24,), horizons=(3,)
    )
    models = forecast.model_features(har_lags=(1, 3, 21), momentum_hours=(24,))
    preds = forecast.walk_forward(
        panel, horizons=(3,), models=models, min_train_months=3, winsor=0.0
    )
    skill = forecast.skill(preds, models=["har"], coefficient=0.4)
    assert skill[0].r2_vs_persistence > 0.2


def test_sharpe_standard_error_matches_the_iid_formula() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(0.001, 0.01, 20000)
    est = stats.sharpe(r, periods_per_year=1.0, coefficient=0.4)
    sr = r.mean() / r.std(ddof=1)
    assert est.se == pytest.approx(np.sqrt((1 + sr**2 / 2) / r.size), rel=0.12)


def test_autocorrelated_returns_widen_the_sharpe_error() -> None:
    rng = np.random.default_rng(5)
    e = rng.normal(0, 0.01, 20001)
    r = 0.0005 + e[1:] + 0.8 * e[:-1]
    iid = np.sqrt((1 + (r.mean() / r.std()) ** 2 / 2) / r.size)
    assert stats.sharpe(r, periods_per_year=1.0, coefficient=0.4).se > 1.3 * iid


@given(trials=st.integers(min_value=2, max_value=500))
def test_expected_max_sharpe_grows_with_trials(trials: int) -> None:
    assert stats.expected_max_sharpe(variance=1.0, trials=trials + 1) > stats.expected_max_sharpe(
        variance=1.0, trials=trials
    )


def test_deflated_sharpe_penalizes_more_trials() -> None:
    few, _ = stats.deflated_sharpe(
        sharpe_per_period=0.1,
        trial_sharpes=np.linspace(-0.05, 0.1, 5),
        periods=500,
        skew=0.0,
        kurtosis=3.0,
    )
    many, _ = stats.deflated_sharpe(
        sharpe_per_period=0.1,
        trial_sharpes=np.linspace(-0.05, 0.1, 200),
        periods=500,
        skew=0.0,
        kurtosis=3.0,
    )
    assert many < few


def test_worst_drawdown_and_recovery() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="1D", tz="UTC")
    equity = pd.Series([1.0, 1.2, 0.9, 1.0, 1.25, 1.3], index=index)
    dd = stats.worst_drawdown(equity)
    assert dd.depth == pytest.approx(0.9 / 1.2 - 1)
    assert dd.recovered == index[4]
    assert dd.duration_days == pytest.approx(3.0)


def test_equal_hazards_split_collateral_evenly() -> None:
    split = counterparty.optimal_split(
        {"a": 0.02, "b": 0.02, "c": 0.02}, lgd=1.0, level=0.95, max_share=1.0
    )
    assert all(w == pytest.approx(1 / 3, abs=1e-6) for w in split.weights.values())
    assert split.expected_shortfall < split.single_venue_shortfall


def test_expected_loss_is_the_weighted_hazard() -> None:
    hazards = {"a": 0.01, "b": 0.05}
    split = counterparty.optimal_split(hazards, lgd=0.5, level=0.9, max_share=1.0)
    expected = sum(split.weights[v] * h * 0.5 for v, h in hazards.items())
    assert split.expected_loss == pytest.approx(expected)


def test_expected_shortfall_of_a_discrete_distribution() -> None:
    es = counterparty.expected_shortfall(np.array([0.0, 1.0]), np.array([0.98, 0.02]), level=0.95)
    assert es == pytest.approx(0.02 / 0.05)


def test_a_tail_wider_than_the_failure_mass_favours_concentration() -> None:
    hazards = {"a": 0.01, "b": 0.015, "c": 0.015}
    wide = counterparty.optimal_split(hazards, lgd=0.5, level=0.95, max_share=1.0)
    assert wide.any_failure_probability < 0.05
    assert max(wide.weights.values()) == pytest.approx(1.0, abs=1e-9)
    narrow = counterparty.optimal_split(hazards, lgd=0.5, level=0.99, max_share=1.0)
    assert max(narrow.weights.values()) < 1.0
    assert narrow.expected_shortfall <= narrow.single_venue_shortfall + 1e-12
    assert narrow.expected_shortfall <= narrow.equal_split_shortfall + 1e-12
    assert all(w >= 0.0 for w in narrow.weights.values())
