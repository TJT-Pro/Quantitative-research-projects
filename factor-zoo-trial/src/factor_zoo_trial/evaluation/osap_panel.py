"""OSAP predictor panel: period labels, factor-adjusted returns and in-sample tests.

Periods come from SignalDoc. A month is in sample when its year lies between
SampleStartYear and SampleEndYear. Publication is placed at the end of the journal
year (configurable month) shifted back by `shift` years; months after it are post
publication, months between the sample end and publication are post sample. Months
before the sample start are dropped, as in McLean and Pontiff (2016).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_zoo_trial.models.inference import ewc_mean, one_sided_p

PERIODS = ("in_sample", "post_sample", "post_publication")
FRENCH_COLUMNS = {"capm": ["mkt_rf"], "ff6": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]}


def label_periods(
    returns: pd.DataFrame, signals: pd.DataFrame, *, publication_month: int, shift: int
) -> pd.DataFrame:
    """Attach a period to each signal-month; drops pre-sample months and undocumented signals."""
    docs = signals[["signal", "year", "sample_start", "sample_end"]].dropna()
    frame = returns.merge(docs, on="signal", how="inner")
    year = frame["month"].dt.year
    month_index = year * 12 + frame["month"].dt.month
    published = (frame["year"] - shift) * 12 + publication_month
    in_sample = (year >= frame["sample_start"]) & (year <= frame["sample_end"])
    after_sample = year > frame["sample_end"]
    after_pub = (month_index > published) & after_sample
    post_sample = after_sample & ~after_pub
    period = np.select([in_sample, post_sample, after_pub], list(PERIODS), default="pre_sample")
    frame = frame.assign(period=period)
    return frame.loc[frame["period"] != "pre_sample", ["signal", "month", "ret", "period"]]


def adjust_returns(returns: pd.DataFrame, french: pd.DataFrame, *, basis: str) -> pd.DataFrame:
    """Raw returns, or alpha plus residual against CAPM or FF5 + momentum (full-sample betas)."""
    if basis == "raw":
        return returns
    columns = FRENCH_COLUMNS[basis]
    merged = returns.merge(french[["month", *columns]].dropna(), on="month", how="inner")
    pieces = []
    for _, group in merged.groupby("signal", sort=False):
        x = group[columns].to_numpy()
        design = np.column_stack([np.ones(len(group)), x])
        coef, *_ = np.linalg.lstsq(design, group["ret"].to_numpy(), rcond=None)
        adjusted = group["ret"].to_numpy() - x @ coef[1:]
        pieces.append(group[["signal", "month"]].assign(ret=adjusted))
    return pd.concat(pieces, ignore_index=True)


def in_sample_tests(panel: pd.DataFrame, *, coefficient: float, min_months: int) -> pd.DataFrame:
    rows = []
    for signal, group in panel.loc[panel["period"] == "in_sample"].groupby("signal"):
        values = group["ret"].to_numpy(dtype=np.float64)
        if values.size < min_months:
            continue
        test = ewc_mean(values, coefficient=coefficient)
        e = values - values.mean()
        rho = float(e[1:] @ e[:-1] / (e @ e))
        rows.append(
            {
                "signal": signal,
                "mean": test.mean,
                "se": test.se,
                "t": test.t,
                "n": test.n,
                "df": test.df,
                "rho1": rho,
            }
        )
    out = pd.DataFrame(rows, columns=["signal", "mean", "se", "t", "n", "df", "rho1"])
    out["p"] = one_sided_p(out["t"].to_numpy(), out["df"].to_numpy(dtype=np.int64))
    return out.sort_values("signal").reset_index(drop=True)


def in_sample_matrix(panel: pd.DataFrame, signals: list[str]) -> np.ndarray:
    """Months x signals of in-sample returns, NaN outside each signal's window."""
    wide = (
        panel.loc[(panel["period"] == "in_sample") & panel["signal"].isin(signals)]
        .pivot_table(index="month", columns="signal", values="ret", aggfunc="first")
        .reindex(columns=signals)
        .sort_index()
    )
    return wide.to_numpy(dtype=np.float64)


def liquidity_share(tests_all: pd.DataFrame, tests_screened: pd.DataFrame) -> pd.Series:
    """Percentile rank of the share of in-sample return that survives the liquidity screen."""
    merged = tests_all[["signal", "mean"]].merge(
        tests_screened[["signal", "mean"]], on="signal", suffixes=("", "_screened")
    )
    share = merged["mean_screened"] / merged["mean"]
    return pd.Series(share.rank(pct=True).to_numpy(), index=merged["signal"], name="liquidity")
