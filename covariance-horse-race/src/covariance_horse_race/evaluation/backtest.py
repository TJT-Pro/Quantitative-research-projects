"""Rolling out-of-sample backtest.

At the last trading day of each month t:

1. Take the trailing `window` trading days ending at t (inclusive).
2. Keep assets with no missing return inside that window. Membership is decided
   from the window alone, never from the holding period.
3. Fit every model on the window and forecast the average daily covariance over
   the next month's trading days.
4. Form unconstrained and long-only GMV weights and hold them fixed through the
   next month (as in Ledoit and Wolf, 2017), recording daily portfolio returns,
   the realized-covariance losses and the turnover needed to rebalance from the
   drifted previous weights.

Holiday schedules are announced in advance, so using the count of next-month
trading days as the forecast horizon does not leak information.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from covariance_horse_race.config import ModelSpec, ResearchConfig
from covariance_horse_race.evaluation import losses
from covariance_horse_race.evaluation.portfolio import (
    SingularForecastError,
    drifted_weights,
    gmv_long_only,
    gmv_weights,
)
from covariance_horse_race.models import CovarianceModel, build_model


def month_end_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer positions of the last trading day of each calendar month."""
    periods = index.to_period("M")
    is_last = np.append(periods[1:] != periods[:-1], True)
    return np.flatnonzero(is_last)


@dataclass
class _Slot:
    """State carried across months for one (window, model) pair."""

    model: CovarianceModel | None = None
    assets: tuple[str, ...] = ()
    previous_weights: dict[str, pd.Series] = field(default_factory=dict)


@dataclass(frozen=True)
class _Month:
    """Everything one rebalance needs, computed once and shared by all models."""

    rebalance_date: pd.Timestamp
    window_dates: pd.DatetimeIndex
    hold_dates: pd.DatetimeIndex
    returns: np.ndarray  # T x N estimation window
    future: np.ndarray  # D x N holding-period returns
    proxy: np.ndarray  # realized covariance proxy of the holding period
    assets: tuple[str, ...]


@dataclass(frozen=True)
class _Settings:
    singular_tolerance: float
    max_iterations: int
    solver_tolerance: float


@dataclass(frozen=True)
class BacktestOutput:
    monthly: pd.DataFrame
    daily: pd.DataFrame


def _factor_window(factors: pd.DataFrame | None, dates: pd.DatetimeIndex) -> np.ndarray | None:
    if factors is None:
        return None
    window = factors.reindex(dates)
    if window.isna().any().any():
        return None
    out: np.ndarray = window.to_numpy(dtype=float)
    return out


def _turnover(new: pd.Series, previous: pd.Series | None) -> float:
    if previous is None:
        return float("nan")
    aligned = pd.concat([new, previous], axis=1).fillna(0.0)
    return float(np.abs(aligned.iloc[:, 0] - aligned.iloc[:, 1]).sum())


def run_universe(
    config: ResearchConfig,
    universe: str,
    returns: pd.DataFrame,
    factor_sets: dict[str, pd.DataFrame],
    progress: Callable[[str], None] | None = None,
) -> BacktestOutput:
    """Backtest every configured (window, model) pair on one universe."""
    bt = config.backtest
    values = returns.to_numpy(dtype=float)
    columns = np.asarray(returns.columns)
    index = pd.DatetimeIndex(returns.index)
    ends = month_end_positions(index)
    longest = max(bt.windows)
    solver = config.portfolio.long_only_solver
    settings = _Settings(bt.singular_tolerance, solver.max_iterations, solver.tolerance)

    slots: dict[tuple[int, str], _Slot] = {}
    monthly: list[dict[str, object]] = []
    daily: list[pd.DataFrame] = []

    for m in range(len(ends) - 1):
        t = int(ends[m])
        hold = slice(t + 1, int(ends[m + 1]) + 1)
        hold_dates = index[hold]
        horizon = len(hold_dates)
        for window in bt.windows:
            first_eligible = (longest if bt.common_evaluation_sample else window) - 1
            if t < first_eligible or t - window + 1 < 0:
                continue
            win = slice(t - window + 1, t + 1)
            block = values[win]
            complete = ~np.isnan(block).any(axis=0)
            if complete.sum() < 2:
                continue
            assets = tuple(str(c) for c in columns[complete])
            future = values[hold][:, complete]
            missing_in_hold = int(np.isnan(future).sum())
            future = np.nan_to_num(future, nan=0.0)
            month = _Month(
                rebalance_date=index[t],
                window_dates=index[win],
                hold_dates=hold_dates,
                returns=np.ascontiguousarray(block[:, complete]),
                future=future,
                proxy=losses.realized_proxy(future),
                assets=assets,
            )
            for spec in config.models:
                if not spec.runs_on(window):
                    continue
                key = (window, spec.name)
                slot = slots.setdefault(key, _Slot())
                if slot.model is None or slot.assets != assets:
                    slot.model = build_model(spec)
                    slot.assets = assets
                record = _evaluate(
                    spec, slot, month=month, factor_sets=factor_sets, settings=settings, daily=daily
                )
                record.update(
                    universe=universe,
                    window=window,
                    model=spec.name,
                    rebalance_date=index[t],
                    n_assets=len(assets),
                    horizon=horizon,
                    missing_in_holding=missing_in_hold,
                )
                monthly.append(record)
        if progress is not None and m % 120 == 0:
            progress(f"{universe}: {index[t].date()}")

    monthly_frame = pd.DataFrame(monthly)
    daily_frame = (
        pd.concat(daily, ignore_index=True)
        if daily
        else pd.DataFrame(
            columns=["date", "rebalance_date", "window", "model", "gmv", "gmv_long_only"]
        )
    )
    daily_frame.insert(0, "universe", universe)
    return BacktestOutput(monthly=monthly_frame, daily=daily_frame)


def _evaluate(
    spec: ModelSpec,
    slot: _Slot,
    *,
    month: _Month,
    factor_sets: dict[str, pd.DataFrame],
    settings: _Settings,
    daily: list[pd.DataFrame],
) -> dict[str, object]:
    """Fit one model for one month and score its forecast; failures are recorded, not raised."""
    nan = float("nan")
    record: dict[str, object] = {
        "status": "ok",
        "gmv_variance": nan,
        "gmv_long_only_variance": nan,
        "qlike": nan,
        "frobenius": nan,
        "turnover": nan,
        "turnover_long_only": nan,
    }
    if slot.model is None:
        raise RuntimeError("model slot was not initialized")
    factors = None
    if slot.model.requires_factors:
        factors = _factor_window(
            factor_sets.get(str(spec.params["factor_set"])), month.window_dates
        )
        if factors is None:
            record["status"] = "no_factors"
            return record
    try:
        slot.model.fit(month.returns, factors)
        forecast = slot.model.forecast(len(month.hold_dates))
    except (ValueError, np.linalg.LinAlgError) as exc:
        record["status"] = f"fit_failed: {exc}"
        slot.model = None
        slot.previous_weights.clear()
        return record
    if not np.isfinite(forecast).all():
        record["status"] = "non_finite"
        slot.previous_weights.clear()
        return record
    record["frobenius"] = losses.frobenius(month.proxy, forecast)
    try:
        w = gmv_weights(forecast, settings.singular_tolerance)
        w_lo = gmv_long_only(
            forecast,
            settings.singular_tolerance,
            settings.max_iterations,
            settings.solver_tolerance,
        )
    except SingularForecastError:
        record["status"] = "singular"
        slot.previous_weights.clear()
        return record
    future = month.future
    record["qlike"] = losses.qlike(month.proxy, forecast)
    record["gmv_variance"] = losses.portfolio_realized_variance(w, future)
    record["gmv_long_only_variance"] = losses.portfolio_realized_variance(w_lo, future)

    index = pd.Index(month.assets)
    new_w, new_lo = pd.Series(w, index=index), pd.Series(w_lo, index=index)
    record["turnover"] = _turnover(new_w, slot.previous_weights.get("gmv"))
    record["turnover_long_only"] = _turnover(new_lo, slot.previous_weights.get("lo"))
    slot.previous_weights["gmv"] = pd.Series(drifted_weights(w, future), index=index)
    slot.previous_weights["lo"] = pd.Series(drifted_weights(w_lo, future), index=index)

    daily.append(
        pd.DataFrame(
            {
                "date": month.hold_dates,
                "rebalance_date": month.rebalance_date,
                "window": int(month.returns.shape[0]),
                "model": spec.name,
                "gmv": future @ w,
                "gmv_long_only": future @ w_lo,
            }
        )
    )
    return record
