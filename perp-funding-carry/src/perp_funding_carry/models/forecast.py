"""Funding forecasts on an 8-hour grid, evaluated walk-forward against persistence.

Prints are summed into 8-hour windows (w - 8h, w], so 1-, 4- and 8-hour contracts share
one unit: funding paid per 8 hours. At window end t the forecaster knows funding up to t,
the basis (mark / spot - 1) at t and spot momentum. The target at horizon h is the mean
funding over windows t+1 ... t+h.

Models: persistence (the last window), HAR (means over the last 1, 3 and 21 windows) and
HAR plus basis and momentum. Pooled OLS is refit at the start of each month on every row
whose target was already realized, after winsorizing with training quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from perp_funding_carry.models.stats import ewc_long_run_variance

GRID = pd.Timedelta(hours=8)


def window_funding(funding: pd.DataFrame) -> pd.DataFrame:
    """symbol, t (window end), f: the sum of prints in (t - 8h, t]."""
    t = (funding["time"] - pd.Timedelta(milliseconds=1)).dt.floor("8h") + GRID
    summed: pd.DataFrame = (
        funding.assign(t=t).groupby(["symbol", "t"], as_index=False).agg(f=("rate", "sum"))
    )
    return summed


def build_panel(
    funding: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    bar_hours: float,
    har_lags: tuple[int, ...],
    momentum_hours: tuple[int, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    win = window_funding(funding)
    close = bars.assign(t=bars["open_time"] + pd.Timedelta(hours=bar_hours))
    close = close.loc[close["t"].dt.hour % 8 == 0, ["symbol", "t", "mark_close", "spot_close"]]
    panel = win.merge(close, on=["symbol", "t"], how="inner").sort_values(["symbol", "t"])
    panel["basis"] = panel["mark_close"] / panel["spot_close"] - 1.0
    pieces = []
    for _, group in panel.groupby("symbol", sort=False):
        g = group.set_index("t").asfreq("8h")  # gaps become NaN rows, so lags never jump them
        for lag in har_lags:
            g[f"har{lag}"] = g["f"].rolling(lag, min_periods=lag).mean()
        log_spot = np.log(g["spot_close"])
        for hours in momentum_hours:
            g[f"mom{hours}"] = log_spot - log_spot.shift(int(hours // 8))
        for h in horizons:
            future = g["f"][::-1].rolling(h, min_periods=h).mean()[::-1].shift(-1)
            g[f"y{h}"] = future
        g["symbol"] = g["symbol"].ffill().bfill()
        pieces.append(g.reset_index())
    out = pd.concat(pieces, ignore_index=True)
    # A feature that is infinite is as unusable as a missing one; let the NaN filters see it.
    numeric = out.select_dtypes("number").columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=["f"]).reset_index(drop=True)


def model_features(
    *, har_lags: tuple[int, ...], momentum_hours: tuple[int, ...], extra: tuple[str, ...] = ()
) -> dict[str, list[str]]:
    har = [f"har{lag}" for lag in har_lags]
    plus = [*har, "basis", *(f"mom{h}" for h in momentum_hours)]
    out = {"har": har, "har_plus": plus}
    if extra:
        out["har_plus_extra"] = [*plus, *extra]
    return out


def _winsor_bounds(
    frame: pd.DataFrame, columns: list[str], q: float
) -> dict[str, tuple[float, float]]:
    return {c: (float(frame[c].quantile(q)), float(frame[c].quantile(1 - q))) for c in columns}


def _clip(frame: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return frame.assign(**{c: frame[c].clip(lo, hi) for c, (lo, hi) in bounds.items()})


def walk_forward(
    panel: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    models: dict[str, list[str]],
    min_train_months: int,
    winsor: float,
) -> pd.DataFrame:
    """Out-of-sample forecasts: symbol, t, horizon, actual, persistence and one column per model."""
    month = panel["t"].dt.tz_localize(None).dt.to_period("M")
    months = sorted(month.unique())
    first = months[0] + min_train_months
    rows = []
    for current in [m for m in months if m >= first]:
        start = pd.Timestamp(current.start_time, tz="UTC")
        test = panel.loc[month == current]
        for h in horizons:
            target = f"y{h}"
            # A training row is usable once its whole target window has settled.
            realized = panel["t"] + GRID * h <= start
            train = panel.loc[realized & (panel["t"] < start)].dropna(subset=[target])
            out = test[["symbol", "t", target, "f"]].rename(
                columns={target: "actual", "f": "persistence"}
            )
            out = out.assign(horizon=h)
            for name, features in models.items():
                fit = train.dropna(subset=features)
                live = test.dropna(subset=features)
                if len(fit) < 10 * (len(features) + 1) or live.empty:
                    out[name] = np.nan
                    continue
                bounds = _winsor_bounds(fit, [*features, target], winsor)
                clipped = _clip(fit, bounds)
                x = np.column_stack([np.ones(len(clipped)), clipped[features].to_numpy()])
                beta, *_ = np.linalg.lstsq(x, clipped[target].to_numpy(), rcond=None)
                live_x = _clip(live, {k: v for k, v in bounds.items() if k != target})
                pred = np.column_stack([np.ones(len(live_x)), live_x[features].to_numpy()]) @ beta
                out[name] = pd.Series(pred, index=live.index).reindex(test.index)
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@dataclass(frozen=True)
class Skill:
    horizon: int
    model: str
    n: int
    r2_vs_persistence: float  # 1 - MSE(model) / MSE(persistence)
    dm_t: float  # Diebold-Mariano t on the per-window mean loss differential
    dm_p: float
    windows: int


def skill(predictions: pd.DataFrame, *, models: list[str], coefficient: float) -> list[Skill]:
    out = []
    for h, group in predictions.groupby("horizon"):
        for model in models:
            g = group.dropna(subset=[model, "actual", "persistence"])
            if len(g) < 30:
                continue
            e_m = (g[model] - g["actual"]) ** 2
            e_p = (g["persistence"] - g["actual"]) ** 2
            r2 = 1.0 - float(e_m.sum() / e_p.sum()) if e_p.sum() > 0 else float("nan")
            # Cross-sectional mean loss differential per window, then a HAR-robust t.
            d = (e_p - e_m).groupby(g["t"]).mean().sort_index().to_numpy()
            if d.size < 10:
                continue
            lrv, nu = ewc_long_run_variance(d, coefficient=coefficient)
            se = np.sqrt(lrv / d.size)
            t_stat = float(d.mean() / se) if se > 0 else 0.0
            out.append(
                Skill(
                    horizon=int(str(h)),
                    model=model,
                    n=len(g),
                    r2_vs_persistence=r2,
                    dm_t=t_stat,
                    dm_p=float(2 * stats.t.sf(abs(t_stat), df=nu)),
                    windows=int(d.size),
                )
            )
    return out
