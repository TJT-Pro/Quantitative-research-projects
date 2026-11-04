"""How much of the long yield's move on policy-rate change days is term premium?

For each day the policy rate changed (after the configured start, when changes
were announced), daily yields on that day and the previous trading day are
decomposed with the model estimated through the end of the previous month, the
model a user actually had that day. The 10-year yield change splits exactly into
the change in expected short rates (risk-neutral yield), the change in the term
premium, and the change in the model's fitting error.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

from acm_term_premium.models.acm import AcmModel


@dataclass(frozen=True)
class EventSummary:
    events: int
    hikes: int
    cuts: int
    mean_abs_yield_bp: float
    mean_abs_expectations_bp: float
    mean_abs_term_premium_bp: float
    term_premium_variance_share: float
    expectations_variance_share: float
    covariance_share: float
    all_days_term_premium_variance_share: float


def _model_for(
    day: pd.Timestamp, models: dict[pd.Timestamp, AcmModel], ends: list[pd.Timestamp]
) -> AcmModel | None:
    """The model estimated through the last month end strictly before `day`'s month."""
    month_start = day.to_period("M").start_time
    i = bisect.bisect_left(ends, month_start)
    return models[ends[i - 1]] if i > 0 else None


def decompose_changes(
    daily_curve: pd.DataFrame,
    days: pd.DatetimeIndex,
    models: dict[pd.Timestamp, AcmModel],
    maturity: int,
) -> pd.DataFrame:
    rows = []
    index = pd.DatetimeIndex(daily_curve.index)
    ends = sorted(models)
    for day in days:
        if day not in index:
            continue
        pos = index.get_loc(day)
        if not isinstance(pos, int) or pos == 0:
            continue
        model = _model_for(day, models, ends)
        if model is None:
            continue
        pair = daily_curve.iloc[[pos - 1, pos]]
        dec = model.decompose(pair, [maturity])
        actual = pair[maturity].to_numpy()
        rn = dec["risk_neutral"][maturity].to_numpy()
        tp = dec["term_premium"][maturity].to_numpy()
        rows.append(
            {
                "date": day,
                "yield_bp": 1e4 * float(actual[1] - actual[0]),
                "expectations_bp": 1e4 * float(rn[1] - rn[0]),
                "term_premium_bp": 1e4 * float(tp[1] - tp[0]),
                "fit_error_bp": 1e4
                * float((actual[1] - actual[0]) - (rn[1] - rn[0]) - (tp[1] - tp[0])),
            }
        )
    return pd.DataFrame(rows)


def _shares(frame: pd.DataFrame) -> tuple[float, float, float]:
    var_y = float(frame["yield_bp"].var())
    var_tp = float(frame["term_premium_bp"].var())
    var_rn = float(frame["expectations_bp"].var())
    cov = float(np.cov(frame["term_premium_bp"], frame["expectations_bp"])[0, 1])
    return var_tp / var_y, var_rn / var_y, 2 * cov / var_y


def summarize(events: pd.DataFrame, changes: pd.DataFrame, all_days: pd.DataFrame) -> EventSummary:
    tp_share, rn_share, cov_share = _shares(events)
    all_tp, _, _ = _shares(all_days)
    merged = events.merge(changes, on="date", how="left")
    return EventSummary(
        events=len(events),
        hikes=int((merged["change_pp"] > 0).sum()),
        cuts=int((merged["change_pp"] < 0).sum()),
        mean_abs_yield_bp=float(events["yield_bp"].abs().mean()),
        mean_abs_expectations_bp=float(events["expectations_bp"].abs().mean()),
        mean_abs_term_premium_bp=float(events["term_premium_bp"].abs().mean()),
        term_premium_variance_share=tp_share,
        expectations_variance_share=rn_share,
        covariance_share=cov_share,
        all_days_term_premium_variance_share=all_tp,
    )


def all_days_summary(all_days: pd.DataFrame) -> dict[str, float]:
    """The same statistics on every trading day of the event sample, as the comparison."""
    tp_share, rn_share, cov_share = _shares(all_days)
    return {
        "days": float(len(all_days)),
        "mean_abs_yield_bp": float(all_days["yield_bp"].abs().mean()),
        "mean_abs_expectations_bp": float(all_days["expectations_bp"].abs().mean()),
        "mean_abs_term_premium_bp": float(all_days["term_premium_bp"].abs().mean()),
        "term_premium_variance_share": tp_share,
        "expectations_variance_share": rn_share,
        "covariance_share": cov_share,
    }
