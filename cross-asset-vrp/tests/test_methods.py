"""The methods against cases whose answer is known in advance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from cross_asset_vrp import config as config_mod
from cross_asset_vrp.models import chain, futures, har, heston, predict, realized, replication
from cross_asset_vrp.plotting import figures


@pytest.fixture(scope="module")
def heston_paths() -> heston.Simulated:
    params = heston.HestonParams(
        kappa=4.0,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        v0=0.04,
        paths=30,
        days=40,
        steps_per_day=1440,
        seed=11,
    )
    return heston.simulate(params, samplings=(1, 5, 30))


def test_range_estimators_recover_integrated_variance_at_fine_sampling(
    heston_paths: heston.Simulated,
) -> None:
    """Unbiased in continuous time; at one-minute sampling of a 24-hour day the discrete range
    costs a few percent, and coarser sampling costs more, never less."""
    truth = heston_paths.integrated.mean()
    for name in ("parkinson", "garman_klass", "rogers_satchell"):
        bias = [
            realized.DAILY_ESTIMATORS[name](heston_paths.bars[step]).mean() / truth - 1.0
            for step in (1, 5, 30)
        ]
        assert bias[0] == pytest.approx(0.0, abs=0.06), name
        assert bias[0] > bias[1] > bias[2], name


def test_coarse_sampling_shrinks_the_range_as_bgk_predict(heston_paths: heston.Simulated) -> None:
    for coarse in (5, 30):
        measured, predicted = heston.range_deficit(heston_paths, fine=1, coarse=coarse)
        assert measured == pytest.approx(predicted, rel=0.15)


def test_trading_and_calendar_day_sums_give_one_variance() -> None:
    """One path observed every calendar day and only on weekdays: the summed variance over the
    same 30 calendar days is the same once each return spans its own elapsed time."""
    rng = np.random.default_rng(3)
    days = pd.date_range("2024-01-01", periods=400, freq="D")
    log_price = np.cumsum(rng.normal(0, 0.01, days.size))
    every = pd.Series(np.diff(log_price, prepend=log_price[0]) ** 2, index=days)
    weekdays = days[days.day_of_week < 5]
    weekday_returns = pd.Series(log_price, index=days).loc[weekdays].diff().fillna(0.0) ** 2
    window = 30
    calendar, _ = har.forward_sum(every, days=window)
    trading, _ = har.forward_sum(weekday_returns, days=window)
    both = pd.DataFrame({"c": calendar, "t": trading}).dropna()
    last_day = pd.DatetimeIndex(both.index).day_of_week
    both = both.loc[last_day < 4]  # windows whose last day is a weekday on both clocks
    ratio = both["t"].sum() / both["c"].sum()
    assert ratio == pytest.approx(1.0, rel=0.05)


def test_har_recovers_its_coefficients() -> None:
    rng = np.random.default_rng(5)
    n = 6000
    beta = (0.00002, 0.35, 0.3, 0.25)
    v = np.full(n, 0.0001)
    for t in range(22, n):
        mean_w, mean_m = v[t - 5 : t].mean(), v[t - 22 : t].mean()
        v[t] = (
            beta[0]
            + beta[1] * v[t - 1]
            + beta[2] * mean_w
            + beta[3] * mean_m
            + rng.normal(0, 0.00001)
        )
    proxy = pd.Series(v)
    x = har.features(proxy, lags=(1, 5, 22)).shift(1)
    ok = x.notna().all(axis=1)
    fit = har.fit(x[ok].to_numpy(), proxy[ok].to_numpy())
    assert fit.coefficients[1:] == pytest.approx(beta[1:], abs=0.05)


def test_har_walk_forward_never_uses_unsettled_targets() -> None:
    days = pd.date_range("2020-01-01", periods=900, freq="D")
    proxy = pd.Series(np.linspace(1, 2, days.size), index=days)
    target, end = har.forward_sum(proxy, days=30)
    out = har.walk_forward(proxy, target, end, lags=(1, 7, 30), min_train_days=100)
    first = out.index.min()
    # The first forecast month needs 100 settled rows: windows that closed before it began.
    assert (end < first.to_period("M").start_time).sum() >= 100


def _flat_chain(
    *, vol: float, spot: float, now: pd.Timestamp, days: tuple[int, ...]
) -> pd.DataFrame:
    rows = []
    for d in days:
        expiry = now + pd.Timedelta(days=d)
        years = d / 365.0
        for k in np.arange(spot * 0.2, spot * 4.0, spot * 0.01):
            s = vol * np.sqrt(years)
            d1 = (np.log(spot / k) + 0.5 * s * s) / s
            call = spot * norm.cdf(d1) - k * norm.cdf(d1 - s)
            put = k * norm.cdf(s - d1) - spot * norm.cdf(-d1)
            for kind, usd in (("call", call), ("put", put)):
                btc = usd / spot
                rows.append(
                    {
                        "type": kind,
                        "strike": k,
                        "expiration": expiry,
                        "bid": btc * 0.99,
                        "ask": btc * 1.01,
                        "forward": spot,
                    }
                )
    return pd.DataFrame(rows)


@settings(deadline=None, max_examples=8)
@given(vol=st.floats(min_value=0.3, max_value=1.2))
def test_replication_returns_a_flat_smile_squared(vol: float) -> None:
    now = pd.Timestamp("2024-01-01", tz="UTC")
    chain = _flat_chain(vol=vol, spot=40_000.0, now=now, days=(14, 45))
    rate = replication.thirty_day_rate(
        chain, now=now, side="mid", target_days=30, min_expiry_days=7, zero_bid_stop=2
    )
    assert rate is not None
    assert rate == pytest.approx(vol**2, rel=0.02)
    bid = replication.thirty_day_rate(
        chain, now=now, side="bid", target_days=30, min_expiry_days=7, zero_bid_stop=2
    )
    ask = replication.thirty_day_rate(
        chain, now=now, side="ask", target_days=30, min_expiry_days=7, zero_bid_stop=2
    )
    assert bid is not None
    assert ask is not None
    assert bid < rate < ask


def test_replication_needs_expiries_on_both_sides_of_30_days() -> None:
    now = pd.Timestamp("2024-01-01", tz="UTC")
    chain = _flat_chain(vol=0.6, spot=40_000.0, now=now, days=(40, 60))
    assert (
        replication.thirty_day_rate(
            chain, now=now, side="mid", target_days=30, min_expiry_days=7, zero_bid_stop=2
        )
        is None
    )


@settings(deadline=None, max_examples=30)
@given(
    vol=st.floats(min_value=0.1, max_value=2.0),
    moneyness=st.floats(min_value=-0.5, max_value=0.5),
    days=st.floats(min_value=1.0, max_value=180.0),
    call=st.booleans(),
)
def test_implied_vol_inverts_black76(vol: float, moneyness: float, days: float, call: bool) -> None:
    forward, years = 50_000.0, days / 365.0
    strike = forward * float(np.exp(moneyness))
    value = chain.black76(forward=forward, strike=strike, years=years, vol=vol, call=call)
    intrinsic = max(forward - strike, 0.0) if call else max(strike - forward, 0.0)
    if value - intrinsic < 1e-6 * forward:  # no time value left to invert
        return
    got = chain.implied_vol(value=value, forward=forward, strike=strike, years=years, call=call)
    assert got == pytest.approx(vol, rel=1e-5)


def test_option_names_parse_to_expiry_strike_and_type() -> None:
    parsed = chain.parse_names(
        pd.Series(["BTC-29MAR24-60000-C", "BTC-5JAN24-42500-P", "BTC-PERPETUAL"])
    )
    assert parsed["expiration"].iloc[0] == pd.Timestamp("2024-03-29 08:00", tz="UTC")
    assert parsed["strike"].iloc[1] == 42_500.0
    assert list(parsed["type"].iloc[:2]) == ["call", "put"]
    assert parsed.iloc[2].isna().all()


def _window(
    *, at: pd.Timestamp, vol: float, basis: float, half: dict[str, float], rng: np.random.Generator
) -> pd.DataFrame:
    """Trades over the 24 hours before `at`: the index drifts up from 60,000 (61,800 at the
    last trade), marks are Black-76 at `vol` on F = basis * index, and prices sit `half` from
    the mark by side."""
    rows: list[dict[str, object]] = []
    for days in (10, 40):
        expiry = at + pd.Timedelta(days=days)
        expiry = expiry.normalize() + pd.Timedelta(hours=8)
        code = f"{expiry.day}{expiry:%b%y}".upper()
        for k in np.arange(30_000, 120_001, 2_500):
            for kind in ("C", "P"):
                # Calls and puts last trade at different moments, so parity must align them.
                for frac in (0.2, 0.6, 0.9) if kind == "C" else (0.1, 0.45, 0.75):
                    moment = at - pd.Timedelta(hours=24 * (1 - frac))
                    spot = 60_000.0 + 2_000.0 * frac
                    f = basis * spot
                    years = (expiry - moment).total_seconds() / chain.SECONDS_PER_YEAR
                    mark = (
                        chain.black76(
                            forward=f, strike=float(k), years=years, vol=vol, call=kind == "C"
                        )
                        / f
                    )
                    if mark < 1e-4:
                        continue
                    bucket = "near" if abs(np.log(k / f)) <= 0.1 else "wing"
                    side = "buy" if rng.random() < 0.5 else "sell"
                    sign = 1.0 if side == "buy" else -1.0
                    rows.append(
                        {
                            "trade_id": str(len(rows)),
                            "instrument_name": f"BTC-{code}-{int(k)}-{kind}",
                            "timestamp": int(moment.value // 1_000_000),
                            "price": mark * (1 + sign * half[bucket]),
                            "mark_price": mark,
                            "index_price": spot,
                            "direction": side,
                        }
                    )
    return pd.DataFrame(rows)


def test_rebuild_recovers_the_forward_the_volatility_and_the_spreads() -> None:
    at = pd.Timestamp("2024-03-01", tz="UTC")
    half = {"near": 0.02, "wing": 0.05}
    trades = _window(at=at, vol=0.55, basis=1.01, half=half, rng=np.random.default_rng(3))
    # A trade after the snapshot with an absurd mark must not reach the chain.
    late = trades.iloc[[0]].assign(
        timestamp=int((at + pd.Timedelta(minutes=1)).value // 1_000_000),
        mark_price=0.9,
        trade_id="late",
    )
    rebuilt = chain.rebuild(
        pd.concat([trades, late]), at=at, spread_buckets=(-0.3, -0.1, 0.1, 0.3), min_spread_trades=5
    )
    assert rebuilt is not None
    c = rebuilt.chain
    assert c["forward"].to_numpy() == pytest.approx(
        1.01 * 61_800.0, rel=1e-6
    )  # index at the last trade
    # Repriced at the snapshot with each option's own volatility, the flat smile replicates.
    rate = replication.thirty_day_rate(
        c.assign(bid=c["mark"], ask=c["mark"]),
        now=at,
        side="mid",
        target_days=30,
        min_expiry_days=7,
        zero_bid_stop=2,
    )
    assert rate == pytest.approx(0.55**2, rel=0.03)
    assert rebuilt.half_spreads["-0.1 to 0.1"] == pytest.approx(0.02, rel=1e-9)
    assert rebuilt.half_spreads["0.1 to 0.3"] == pytest.approx(0.05, rel=1e-9)
    assert (c["age_minutes"] >= 0).all()


def test_rebuild_needs_both_sides_near_the_money() -> None:
    at = pd.Timestamp("2024-03-01", tz="UTC")
    trades = _window(
        at=at, vol=0.5, basis=1.0, half={"near": 0.01, "wing": 0.01}, rng=np.random.default_rng(4)
    )
    calls_only = trades.loc[trades["instrument_name"].str.endswith("-C")]
    assert chain.rebuild(calls_only, at=at, spread_buckets=(-0.1, 0.1), min_spread_trades=5) is None


def test_realized_kernel_removes_iid_noise() -> None:
    rng = np.random.default_rng(9)
    n = 1440
    efficient = np.cumsum(rng.normal(0, 0.02 / np.sqrt(n), n + 1))
    observed = efficient + rng.normal(0, 0.0004, n + 1)
    truth = float(np.sum(np.diff(efficient) ** 2))
    noisy = np.diff(observed)
    assert realized.realized_variance(noisy) > 1.3 * truth
    assert realized.realized_kernel(noisy) == pytest.approx(truth, rel=0.2)


def test_constant_maturity_weights_hit_30_days_and_roll() -> None:
    days = pd.bdate_range("2024-01-02", "2024-03-29")
    e1, e2, e3 = pd.Timestamp("2024-01-17"), pd.Timestamp("2024-02-14"), pd.Timestamp("2024-03-20")
    rows = []
    for e in (e1, e2, e3):
        for d in days[days <= e]:
            rows.append({"trade_date": d, "expiry": e, "settle": 20.0})
    cm = futures.constant_maturity(pd.DataFrame(rows), target_days=30, cost_points=0.0)
    flat = cm["ret"].dropna()
    assert np.allclose(flat, 0.0)  # no price change, no cost: no return
    with_cost = futures.constant_maturity(pd.DataFrame(rows), target_days=30, cost_points=0.05)
    assert (with_cost["ret"].dropna() <= 0).all()
    assert with_cost["ret"].dropna().lt(0).any()  # rolling trades weight every day


def test_sizing_uses_only_the_past_and_respects_the_cap() -> None:
    signal = pd.Series(np.r_[np.ones(300), np.full(10, 10.0), -np.ones(10)])
    size = futures.expanding_scale(signal, min_history=250, cap=2.0, positive_only=True)
    assert size.iloc[:250].isna().all()
    assert size.max() == pytest.approx(2.0)
    assert size.iloc[-1] == 0.0
    assert size.iloc[300] == pytest.approx(1.0)  # the jump on day 300 is only seen on day 301


def test_ewc_regression_finds_a_planted_slope() -> None:
    rng = np.random.default_rng(1)
    n = 600
    x = rng.standard_normal(n)
    y = 0.5 * x + rng.standard_normal(n)
    reg = predict.ols_ewc(
        x, y, coefficient=0.4, index=pd.date_range("2000-01-31", periods=n, freq="ME")
    )
    assert reg is not None
    assert reg.ci_low < 0.5 < reg.ci_high
    assert reg.t > 5


def test_forward_excess_sums_the_next_months() -> None:
    monthly = pd.DataFrame({"ret": [0.01, 0.02, 0.03, 0.04], "rf": [0.0, 0.0, 0.0, 0.0]})
    out = predict.forward_excess(monthly, months=2)
    assert out.iloc[0] == pytest.approx((0.02 + 0.03) * 6)
    assert np.isnan(out.iloc[-1])


def test_walk_forward_without_enough_history_returns_numbers() -> None:
    days = pd.date_range("2024-01-01", periods=60, freq="D")
    proxy = pd.Series(1.0, index=days)
    target, end = har.forward_sum(proxy, days=30)
    out = har.walk_forward(proxy, target, end, lags=(1, 7, 30), min_train_days=1000)
    assert out.empty
    assert out["forecast"].dtype == float


def test_an_account_cannot_lose_more_than_its_equity() -> None:
    days = pd.date_range("2018-02-01", periods=6, freq="B")
    sized = pd.Series([0.01, 0.02, -1.6, 0.05, -0.02, 0.03], index=days)
    capped, ruin = futures.limited_liability(sized)
    assert ruin == days[2]
    assert capped.tolist() == [0.01, 0.02, -1.0, 0.0, 0.0, 0.0]
    assert np.prod(1.0 + capped.to_numpy(dtype=np.float64)) == 0.0


@given(st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=1, max_size=60))
@settings(max_examples=200, deadline=None)
def test_a_solvent_account_is_unchanged(returns: list[float]) -> None:
    r = pd.Series(returns, index=pd.date_range("2020-01-01", periods=len(returns), freq="B"))
    capped, ruin = futures.limited_liability(r)
    assert ruin is None
    pd.testing.assert_series_equal(capped, r.astype(float))


def test_futures_figure_marks_only_a_real_wipe_out(tmp_path: Path) -> None:
    days = pd.bdate_range("2012-01-02", "2019-12-31")
    path = pd.Series(np.linspace(1.0, 2.0, len(days)), index=days)
    wiped = path.where(days < pd.Timestamp("2018-02-05"), 0.0)
    gappy = path.drop(days[800])  # a day this rule lacks is NaN in the shared table
    pd.DataFrame({"unconditional": path, "premium": wiped, "slope": gappy}).to_csv(
        tmp_path / "futures_equity.csv", index_label="date"
    )
    cfg = config_mod.load(path=Path(__file__).parents[1] / "configs" / "research.yaml")
    fig = figures.futures_equity(tmp_path, cfg, cfg.plot.palette)
    marks = [[t.get_text() for t in ax.texts] for ax in fig.axes]
    plt.close(fig)
    assert marks == [[], ["account lost 2018-02-05"], []]
