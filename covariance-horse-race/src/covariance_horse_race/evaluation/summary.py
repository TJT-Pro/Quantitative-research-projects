"""Aggregation of backtest output into comparison tables and MCS results.

Comparisons are always made on a common sample: for a given universe, window and
period, a model enters only if it produced a valid forecast in every month of
that period, and only those months are used. A model can drop out for two
reasons, both reported rather than hidden: its forecast is singular (the sample
covariance when assets outnumber observations), or its inputs do not exist yet
(the extended factor set starts in 1963).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from covariance_horse_race.config import ResearchConfig
from covariance_horse_race.evaluation.mcs import McsResult, model_confidence_set


@dataclass(frozen=True)
class Comparison:
    universe: str
    window: int
    period: str
    months: list[pd.Timestamp]
    models: list[str]
    excluded: dict[str, str]


def _period_label(date: pd.Timestamp, period: str) -> str:
    if period == "full":
        return "full"
    decade = (date.year // 10) * 10
    return f"{decade}s"


_STATUS_TEXT = {
    "no_factors": "factor data not yet available",
    "singular": "forecast covariance is singular",
    "non_finite": "forecast has non-finite entries",
    "missing": "no forecast in some months",
}


def describe_status(status: str) -> str:
    """Reader-facing text for a backtest status code."""
    if status.startswith("fit_failed"):
        return "estimation failed" + status.removeprefix("fit_failed")
    return _STATUS_TEXT.get(status, status)


def comparisons(monthly: pd.DataFrame, period: str) -> list[Comparison]:
    """Common-sample comparison sets for every (universe, window, period)."""
    out: list[Comparison] = []
    frame = monthly.copy()
    frame["period"] = [_period_label(pd.Timestamp(d), period) for d in frame["rebalance_date"]]
    for (universe, window, label), group in frame.groupby(["universe", "window", "period"]):
        all_months = sorted(group["rebalance_date"].unique())
        models: list[str] = []
        excluded: dict[str, str] = {}
        for model, rows in group.groupby("model"):
            ok = rows[rows["status"] == "ok"]
            if len(ok) == len(all_months):
                models.append(str(model))
            else:
                statuses = rows.loc[rows["status"] != "ok", "status"].value_counts()
                status = str(statuses.index[0]) if len(statuses) else "missing"
                first_ok = ok["rebalance_date"].min()
                note = (
                    f"{describe_status(status)}: defined in {len(ok)} of {len(all_months)} months"
                )
                if pd.notna(first_ok):
                    note += f", first on {pd.Timestamp(first_ok).date()}"
                excluded[str(model)] = note
        out.append(
            Comparison(
                universe=str(universe),
                window=int(str(window)),
                period=str(label),
                months=[pd.Timestamp(m) for m in all_months],
                models=sorted(models),
                excluded=excluded,
            )
        )
    return out


def loss_matrix(monthly: pd.DataFrame, comparison: Comparison, loss: str) -> pd.DataFrame:
    rows = monthly[
        (monthly["universe"] == comparison.universe)
        & (monthly["window"] == comparison.window)
        & (monthly["rebalance_date"].isin(comparison.months))
        & (monthly["model"].isin(comparison.models))
    ]
    if rows.duplicated(["rebalance_date", "model"]).any():
        raise ValueError("duplicate model-month rows in backtest output")
    matrix = rows.set_index(["rebalance_date", "model"])[loss].unstack("model")  # noqa: PD010
    return matrix[comparison.models].sort_index()


def performance_table(
    monthly: pd.DataFrame, daily: pd.DataFrame, comparison: Comparison, annualization: int
) -> pd.DataFrame:
    """Annualized out-of-sample volatility, turnover and mean losses on the common sample."""
    months = set(comparison.months)
    m_rows = monthly[
        (monthly["universe"] == comparison.universe)
        & (monthly["window"] == comparison.window)
        & (monthly["model"].isin(comparison.models))
        & (monthly["rebalance_date"].isin(months))
    ]
    if not months:
        return pd.DataFrame()
    # Every daily return carries the rebalance date whose weights produced it.
    d_rows = daily[
        (daily["universe"] == comparison.universe)
        & (daily["window"] == comparison.window)
        & (daily["model"].isin(comparison.models))
        & (daily["rebalance_date"].isin(months))
    ]

    table = pd.DataFrame(index=pd.Index(comparison.models, name="model"))
    scale = np.sqrt(annualization)
    table["vol_gmv"] = d_rows.groupby("model")["gmv"].std() * scale
    table["vol_gmv_long_only"] = d_rows.groupby("model")["gmv_long_only"].std() * scale
    grouped = m_rows.groupby("model")
    table["turnover"] = grouped["turnover"].mean()
    table["turnover_long_only"] = grouped["turnover_long_only"].mean()
    table["qlike"] = grouped["qlike"].mean()
    table["frobenius"] = grouped["frobenius"].mean()
    table["mean_assets"] = grouped["n_assets"].mean()
    table["months"] = len(comparison.months)
    table["assets_per_observation"] = table["mean_assets"] / comparison.window
    return table.sort_values("vol_gmv")


@dataclass(frozen=True)
class McsRecord:
    comparison: Comparison
    loss: str
    result: McsResult


def run_mcs(
    config: ResearchConfig, monthly: pd.DataFrame, losses: Sequence[str]
) -> list[McsRecord]:
    settings = config.evaluation.mcs
    rng = np.random.default_rng(config.evaluation.seed)
    records: list[McsRecord] = []
    for comparison in comparisons(monthly, settings.period):
        if comparison.window != config.backtest.primary_window:
            continue
        if len(comparison.months) < settings.min_months or len(comparison.models) < 2:
            continue
        for loss in losses:
            matrix = loss_matrix(monthly, comparison, loss)
            result = model_confidence_set(
                matrix.to_numpy(dtype=float),
                list(matrix.columns),
                alpha=settings.alpha,
                replications=settings.bootstrap_replications,
                mean_block_length=settings.mean_block_length,
                rng=rng,
            )
            records.append(McsRecord(comparison=comparison, loss=loss, result=result))
    return records
