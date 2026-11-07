"""The panel exactly as it was published on a given day.

ALFRED gives each value with the interval of days during which it was the published one.
The panel on day d keeps, for every series and period, the value whose interval contains d,
so a value published after d is never used: `levels_on` checks this on every call and
raises if it would happen.

Periods live on one monthly grid. A quarterly value sits on the third month of its quarter,
as in the reference code, so the quarterly aggregation weights line up with the factor lags.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from realtime_nowcast.config import SeriesConfig

Array = NDArray[np.float64]
FOREVER = "9999-12-31"
DAY = np.timedelta64(1, "D")


class LeakError(AssertionError):
    """A value published after the panel's day would have been used."""


@dataclass(frozen=True)
class SeriesVintages:
    """One series: every value with the period it measures and the days it was current."""

    id: str
    frequency: str
    period: NDArray[np.int64]  # month index on the grid (months since 1970-01)
    start: NDArray[np.datetime64]  # first day the value was the published one
    end: NDArray[np.datetime64]  # last day (inclusive)
    value: Array  # NaN where ALFRED shows "."


def month_index(period: pd.Period) -> int:
    return int(period.year * 12 + period.month - 1 - 1970 * 12)


def index_month(index: int) -> pd.Period:
    return pd.Period(year=1970 + index // 12, month=index % 12 + 1, freq="M")


def from_rows(spec: SeriesConfig, rows: pd.DataFrame) -> SeriesVintages:
    """`rows`: realtime_start, realtime_end, date, value as ALFRED writes them."""
    dates = pd.to_datetime(rows["date"], format="%Y-%m-%d")
    months = np.asarray(
        (dates.dt.year * 12 + dates.dt.month - 1 - 1970 * 12).to_numpy(), dtype=np.int64
    )
    if spec.frequency == "q":
        if not np.all((dates.dt.month - 1) % 3 == 0):
            raise ValueError(f"{spec.id}: quarterly dates must open a quarter")
        months = months + 2  # the third month of the quarter
    end = rows["realtime_end"].replace(FOREVER, "2262-04-11")  # the last day numpy can hold
    value = pd.to_numeric(rows["value"].replace(".", np.nan), errors="raise")
    order = np.lexsort((pd.to_datetime(rows["realtime_start"]).to_numpy(), months))
    return SeriesVintages(
        id=spec.id,
        frequency=spec.frequency,
        period=months[order],
        start=pd.to_datetime(rows["realtime_start"]).to_numpy().astype("datetime64[D]")[order],
        end=pd.to_datetime(end).to_numpy().astype("datetime64[D]")[order],
        value=value.to_numpy(dtype=np.float64)[order],
    )


def first_release(v: SeriesVintages) -> pd.DataFrame:
    """period, release (the first day a value existed for it), value."""
    seen = ~np.isnan(v.value)
    frame = pd.DataFrame({"period": v.period[seen], "start": v.start[seen], "value": v.value[seen]})
    first = frame.sort_values(["period", "start"]).groupby("period", as_index=False).first()
    return first.rename(columns={"start": "release"})


@dataclass(frozen=True)
class Panel:
    first: int  # month index of row 0
    levels: Array  # (T, N) levels as published on `day`
    day: np.datetime64


def levels_on(
    series: tuple[SeriesVintages, ...], *, day: np.datetime64, first: int, last: int
) -> Panel:
    """Levels on the grid first..last (month indexes, inclusive) as published on `day`."""
    t_len = last - first + 1
    out = np.full((t_len, len(series)), np.nan)
    published = np.full((t_len, len(series)), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    for j, v in enumerate(series):
        live = (v.start <= day) & (v.end >= day)
        rows = v.period[live] - first
        keep = (rows >= 0) & (rows < t_len)
        out[rows[keep], j] = v.value[live][keep]
        published[rows[keep], j] = v.start[live][keep]
    used = published[~np.isnan(out)]
    if used.size and used.max() > day:
        raise LeakError(f"a value published on {used.max()} reached the panel of {day}")
    return Panel(first=first, levels=out, day=day)


def transform(levels: Array, specs: tuple[SeriesConfig, ...]) -> Array:
    """The reference transformations: chg, pch (percent), pca (annualized percent on the
    quarterly rows) and lin. Monthly series step one month, quarterly series three."""
    out = np.full_like(levels, np.nan)
    for j, s in enumerate(specs):
        x = levels[:, j]
        step = 1 if s.frequency == "m" else 3
        now, before = x[step:], x[:-step]
        if s.transform == "lin":
            out[:, j] = x
        elif s.transform == "chg":
            out[step:, j] = now - before
        elif s.transform == "pch":
            out[step:, j] = 100.0 * (now / before - 1.0)
        else:  # pca
            out[step:, j] = 100.0 * ((now / before) ** (12 / step) - 1.0)
    return out


def release_days(
    series: tuple[SeriesVintages, ...], *, after: np.datetime64
) -> NDArray[np.datetime64]:
    """Every day after `after` on which some value in the panel changed."""
    days = np.concatenate([v.start[v.start > after] for v in series])
    return np.unique(days)
