"""Model inputs built from the parsed sources.

* Monthly zero curve: the last available day of every complete month, at
  maturities 1..max_maturity months, from the Svensson parameters.
* Short-end repair: optionally replace the one-month yield before a cutoff by
  its fitted value from a regression on the effective fed funds rate estimated
  on later months.
* Daily curve: the same maturities every day, for the policy-day analysis.
* Policy-change days: days on which the federal funds target (or the upper end
  of the target range) differs from the previous observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from acm_term_premium.config import ResearchConfig
from acm_term_premium.data.parsers import parse_acm_workbook, parse_fred, parse_gsw, svensson_yields
from acm_term_premium.data.source import SourceError, verify_raw


def complete_month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last observed day of each month, dropping a trailing month still in progress."""
    days = pd.Series(index, index=index)
    last = days.groupby(index.to_period("M")).max()
    ends = pd.DatetimeIndex(last.to_numpy())
    latest = ends[-1]
    if latest < latest + pd.offsets.BMonthEnd(0):
        ends = ends[:-1]
    return ends


@dataclass(frozen=True)
class SpliceReport:
    intercept: float
    slope: float
    fit_months: int
    replaced_months: int
    r_squared: float


def splice_short_rate(
    curve: pd.DataFrame, fedfunds: pd.Series, fit_start: str, replace_before: str
) -> tuple[pd.DataFrame, SpliceReport]:
    """Replace early one-month yields by their fitted relation to fed funds."""
    one_month = curve[1].copy()
    one_month.index = pd.DatetimeIndex(one_month.index).to_period("M")
    ff = fedfunds.copy() / 100.0
    ff.index = pd.DatetimeIndex(ff.index).to_period("M")
    common = one_month.index.intersection(ff.index)
    fit = common[common >= pd.Period(fit_start, "M")]
    if len(fit) < 24:
        raise SourceError("fewer than 24 months to fit the short-rate splice")
    x = np.column_stack([np.ones(len(fit)), ff.loc[fit].to_numpy()])
    y = one_month.loc[fit].to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    replace = one_month.index[one_month.index < pd.Period(replace_before, "M")]
    replace = replace.intersection(ff.index)
    out = curve.copy()
    rows = pd.DatetimeIndex(curve.index)[pd.DatetimeIndex(curve.index).to_period("M").isin(replace)]
    out.loc[rows, 1] = beta[0] + beta[1] * ff.loc[replace].to_numpy()
    return out, SpliceReport(float(beta[0]), float(beta[1]), len(fit), len(replace), r2)


def policy_change_days(target: pd.Series, target_upper: pd.Series) -> pd.DataFrame:
    """Days the policy rate moved, with the size of the move in percentage points.

    The two series are joined where the range replaces the point target; the
    switch itself is not counted as a policy change.
    """
    switch = target_upper.index[0]
    joined = pd.concat([target[target.index < switch], target_upper])
    change = joined.diff()
    moved = change[change.abs() > 1e-9]
    moved = moved[moved.index != switch]
    return pd.DataFrame({"date": moved.index, "change_pp": moved.to_numpy()})


def build(
    config: ResearchConfig, raw: Path, processed: Path, manifest: Path
) -> SpliceReport | None:
    verify_raw(raw, manifest)
    processed.mkdir(parents=True, exist_ok=True)
    sources = config.data.sources
    params = parse_gsw((raw / sources.gsw.file).read_text(encoding="utf-8"))
    months = np.arange(1, config.model.max_maturity + 1)
    daily = svensson_yields(params, months)
    daily.columns = daily.columns.astype(str)
    daily.to_parquet(processed / "curve_daily.parquet")

    ends = complete_month_ends(pd.DatetimeIndex(params.index))
    monthly = svensson_yields(params.loc[ends], months)
    raw_monthly = monthly.copy()
    report = None
    if config.model.short_rate.method == "fedfunds_splice":
        fedfunds = parse_fred((raw / sources.fedfunds.file).read_text(encoding="utf-8"), "FEDFUNDS")
        monthly, report = splice_short_rate(
            monthly,
            fedfunds,
            config.model.short_rate.fit_start,
            config.model.short_rate.replace_before,
        )
    for frame, name in ((monthly, "curve_monthly"), (raw_monthly, "curve_monthly_unspliced")):
        out = frame.copy()
        out.columns = out.columns.astype(str)
        out.to_parquet(processed / f"{name}.parquet")

    target = parse_fred((raw / sources.target.file).read_text(encoding="utf-8"), "DFEDTAR")
    upper = parse_fred((raw / sources.target_upper.file).read_text(encoding="utf-8"), "DFEDTARU")
    policy_change_days(target, upper).to_parquet(processed / "policy_changes.parquet")

    for sheet, name in (("ACM Monthly", "published_monthly"), ("ACM Daily", "published_daily")):
        parse_acm_workbook(raw / sources.acm.file, sheet).to_parquet(processed / f"{name}.parquet")
    return report


def load_curve(processed: Path, name: str) -> pd.DataFrame:
    path = processed / f"{name}.parquet"
    if not path.is_file():
        raise SourceError(f"{path.name} not built; run `acm build`")
    frame = pd.read_parquet(path)
    frame.columns = frame.columns.astype(int)
    return frame


def load_table(processed: Path, name: str) -> pd.DataFrame:
    path = processed / f"{name}.parquet"
    if not path.is_file():
        raise SourceError(f"{path.name} not built; run `acm build`")
    return pd.read_parquet(path)
