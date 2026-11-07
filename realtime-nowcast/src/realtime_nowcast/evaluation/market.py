"""Do the model's news surprises move Treasury yields on release days?

On each day some panel series was released, the news is the part of that day's change in
the current-quarter nowcast that the model attributes to the releases (revisions and
re-estimation excluded), in percentage points of annualized GDP growth. The daily change in
each yield, in basis points, is regressed on it with heteroskedasticity-robust (HC3)
standard errors: a positive slope says the market prices the information the model
extracts. Release days are business days with a yield on that day and the one before.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from realtime_nowcast.config import Config


def release_news(changes: pd.DataFrame) -> pd.DataFrame:
    """date, news: release news for the quarter that contains the day."""
    upd = changes.loc[changes["kind"] == "update"].copy()
    upd = upd.loc[pd.PeriodIndex(upd["date"], freq="Q").astype(str) == upd["quarter"]]
    out = pd.DataFrame(upd.groupby("date", as_index=False)["news"].sum())
    return pd.DataFrame(out.loc[out["news"] != 0.0])


def yield_changes(yields: pd.DataFrame, series: str) -> pd.DataFrame:
    """date, change (basis points) from the previous business day with a yield."""
    y = yields.loc[yields["series"] == series].sort_values("date")
    change = y["value"].diff() * 100.0
    return pd.DataFrame({"date": y["date"].to_numpy(), "change": change.to_numpy()}).dropna()


def ols_hc3(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.column_stack([np.ones_like(x), x])
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    meat = design.T @ (design * (resid / (1 - leverage))[:, None] ** 2)
    cov = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(cov[1, 1]))
    t = float(beta[1] / se)
    n = y.size
    tss = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(beta[1]),
        "intercept": float(beta[0]),
        "se": se,
        "t": t,
        "p": float(2 * stats.t.sf(abs(t), df=n - 2)),
        "ci_low": float(beta[1] - stats.t.ppf(0.975, n - 2) * se),
        "ci_high": float(beta[1] + stats.t.ppf(0.975, n - 2) * se),
        "r2": float(1 - np.sum(resid**2) / tss) if tss > 0 else float("nan"),
        "n": int(n),
    }


def deciles(frame: pd.DataFrame) -> list[dict[str, float]]:
    """Mean yield change by decile of news, with a 95% interval, for the figure."""
    group = pd.qcut(frame["news"], 10, labels=False, duplicates="drop")
    out = []
    for _, g in frame.groupby(group):
        n = g.shape[0]
        sd = float(g["change"].std(ddof=1)) if n > 1 else float("nan")
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
        out.append(
            {
                "news": float(g["news"].mean()),
                "change": float(g["change"].mean()),
                "half_width": half,
                "n": n,
            }
        )
    return out


def test(config: Config, changes: pd.DataFrame, yields: pd.DataFrame) -> dict[str, Any]:
    news = release_news(changes)
    out: dict[str, Any] = {}
    excluded = list(config.evaluation.excluded_years)
    for series in config.market.yields:
        frame = news.merge(yield_changes(yields, series), on="date", how="inner")
        keep = ~frame["date"].dt.year.isin(excluded)
        out[series] = {
            "all": ols_hc3(frame["news"].to_numpy(), frame["change"].to_numpy()),
            "ex_excluded": ols_hc3(
                frame.loc[keep, "news"].to_numpy(), frame.loc[keep, "change"].to_numpy()
            ),
            "deciles": deciles(frame),
            "first_day": str(frame["date"].min().date()),
            "last_day": str(frame["date"].max().date()),
        }
    return out
