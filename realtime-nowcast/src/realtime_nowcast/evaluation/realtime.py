"""Run the model through history exactly as it could have been run on each day.

For every quarter from the first GDPNow quarter on, the nowcast is tracked from the first
day of the quarter to the day before its advance estimate. The days on which anything can
change are the days some panel value was published or revised, the first day of each month
(when the parameters are re-estimated on that day's vintage) and the first day of each
quarter (when a new quarter is first nowcast). On each such day:

1. the panel is rebuilt from the values published by that day (vintages.levels_on);
2. with the parameters in force, the change since the previous such day is split into
   revisions and the news of each release (models.news), for every quarter being tracked;
3. on the first day of a month EM re-estimates the parameters on that vintage, starting
   from the previous estimates, and the change this makes to each nowcast is recorded as
   its own cause.

Every change in a nowcast therefore has a named cause, and the parts add up to the change:
the identity is checked on every day and the run stops if it fails.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from realtime_nowcast.config import Config
from realtime_nowcast.data import vintages
from realtime_nowcast.models import dfm, news

Array = NDArray[np.float64]
Say = Callable[[str], object]
IDENTITY_TOLERANCE = 1e-7  # percentage points; the decomposition is exact up to rounding
MIN_OBSERVATIONS = 24  # two years of published values before a series enters the model
MIN_BLOCK_SERIES = 2
EM_REPORT_EVERY = 250  # a long EM run prints a line every this many iterations


@dataclass(frozen=True)
class History:
    nowcasts: pd.DataFrame  # date, quarter, nowcast (after every effect of that day)
    changes: pd.DataFrame  # date, quarter, kind, before, revisions, news, reestimation, after
    releases: pd.DataFrame  # date, quarter, series, period, actual, forecast, weight, impact
    estimations: pd.DataFrame  # date, iterations, converged, loglik, seconds, scales, rho

    def names(self) -> dict[str, pd.DataFrame]:
        return {
            "nowcasts": self.nowcasts,
            "changes": self.changes,
            "releases": self.releases,
            "estimations": self.estimations,
        }


def structure_of(config: Config, columns: tuple[int, ...] | None = None) -> dfm.Structure:
    """The block structure of the panel, or of the series in `columns` (indexes into the
    ordered panel)."""
    ordered = config.panel.ordered
    specs = ordered if columns is None else tuple(ordered[i] for i in columns)
    blocks = np.array(
        [[int(b in s.blocks) for b in config.panel.blocks] for s in specs], dtype=np.int64
    )
    monthly = np.array([s.frequency == "m" for s in specs])
    # A block's factor is told apart from its series' own components only with two or more
    # monthly series; with fewer the block is left out (the global block always stays).
    used = blocks[monthly].sum(axis=0) >= MIN_BLOCK_SERIES
    used[0] = True
    return dfm.Structure(
        n_monthly=sum(s.frequency == "m" for s in specs),
        n_quarterly=sum(s.frequency == "q" for s in specs),
        blocks=blocks[:, used],
    )


def settings_of(config: Config) -> dfm.Settings:
    m = config.model
    return dfm.Settings(
        threshold=m.threshold,
        max_iter=m.max_iter,
        measurement_variance=m.measurement_variance,
        pandemic_free=tuple(pd.Period(p, freq="M") for p in m.pandemic_free),
        rho_start=m.rho_start,
        accelerate=m.accelerate,
    )


@dataclass(frozen=True)
class Window:
    quarter: pd.Period
    start: np.datetime64  # first day tracked
    release: np.datetime64  # the advance estimate; tracked through the day before

    def active(self, day: np.datetime64) -> bool:
        return bool(self.start <= day < self.release)


def windows(config: Config, advance: pd.DataFrame, gdpnow: pd.DataFrame) -> list[Window]:
    """Quarters from the first GDPNow quarter that have both an advance estimate and GDPNow
    nowcasts."""
    first = pd.Period(config.evaluation.first_quarter, freq="Q")
    covered = set(gdpnow["quarter"])
    out = []
    for quarter, release in zip(advance["quarter"], advance["release"], strict=True):
        q = pd.Period(quarter, freq="Q")
        if q < first or quarter not in covered:
            continue
        out.append(
            Window(
                quarter=q,
                start=np.datetime64(q.start_time.date(), "D"),
                release=np.datetime64(pd.Timestamp(release).date(), "D"),
            )
        )
    return sorted(out, key=lambda w: w.quarter)


def compute_days(
    series: tuple[vintages.SeriesVintages, ...], spans: list[Window]
) -> tuple[NDArray[np.datetime64], set[np.datetime64]]:
    """Every day a nowcast can change, and the subset that are re-estimation days."""
    start = min(w.start for w in spans)
    end = max(w.release for w in spans) - np.timedelta64(1, "D")
    months = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")
    estimation = {np.datetime64(d.date(), "D") for d in months}
    released = vintages.release_days(series, after=start - np.timedelta64(1, "D"))
    released = released[released <= end]
    entries = {w.start for w in spans}
    days = np.unique(
        np.concatenate([released, np.array(sorted(estimation | entries), dtype="datetime64[D]")])
    )
    return days, estimation


def panel(
    series: tuple[vintages.SeriesVintages, ...], config: Config, *, day: np.datetime64, last: int
) -> tuple[Array, pd.PeriodIndex]:
    first_level = vintages.month_index(pd.Period(config.panel.observation_start, freq="M"))
    first_row = vintages.month_index(pd.Period(config.panel.estimation_start, freq="M"))
    panel = vintages.levels_on(series, day=day, first=first_level, last=last)
    x = vintages.transform(panel.levels, config.panel.ordered)[first_row - first_level :]
    periods = pd.period_range(vintages.index_month(first_row), periods=x.shape[0], freq="M")
    return x, periods


def available(x: Array, *, target: int) -> tuple[int, ...]:
    """Series with at least MIN_OBSERVATIONS published values in the panel. ALFRED's record of
    some series starts years after the series itself (the Philadelphia survey's in 2015), and
    a series enters the model only once that record does. The target must be there."""
    counts = np.sum(~np.isnan(x), axis=0)
    if counts[target] < MIN_OBSERVATIONS:
        raise ValueError("the target has too few published values to estimate the model")
    return tuple(int(i) for i in np.flatnonzero(counts >= MIN_OBSERVATIONS))


def _pad(x: Array, rows: int) -> Array:
    if x.shape[0] >= rows:
        return x[:rows]
    return np.vstack([x, np.full((rows - x.shape[0], x.shape[1]), np.nan)])


RELEASE_COLUMNS = ["date", "quarter", "series", "period", "actual", "forecast", "weight", "impact"]


class _Runner:
    """Carries the parameters, the previous day's panel and the tracked nowcasts across days,
    and collects the output rows."""

    def __init__(
        self, config: Config, series: tuple[vintages.SeriesVintages, ...], say: Say
    ) -> None:
        self.config = config
        self.series = series
        self.say = say
        self.settings = settings_of(config)
        self.ids = [s.id for s in config.panel.ordered]
        self.target = self.ids.index(config.panel.target)
        self.first_row = vintages.month_index(pd.Period(config.panel.estimation_start, freq="M"))
        self.params: dfm.Params | None = None
        self.columns: tuple[int, ...] = ()
        self.previous: Array | None = None
        self.nowcast: dict[pd.Period, float] = {}
        self.nowcasts: list[dict[str, object]] = []
        self.changes: list[dict[str, object]] = []
        self.releases: list[dict[str, object]] = []
        self.estimations: list[dict[str, object]] = []

    @property
    def target_column(self) -> int:
        """The target's position among the series in the model."""
        return self.columns.index(self.target)

    def row_of(self, quarter: pd.Period) -> int:
        return vintages.month_index(quarter.asfreq("M", "end")) - self.first_row

    def day(self, day: np.datetime64, active: list[Window], *, reestimate: bool) -> None:
        last = max(vintages.month_index(w.quarter.asfreq("M", "end")) for w in active)
        x, periods = panel(self.series, self.config, day=day, last=last)
        stamp = pd.Timestamp(day)
        fresh = self.params is None
        if self.params is None:
            self.columns = available(x, target=self.target)
            self.params = self._estimate(x, periods, start=None, day=day)
        self._update(x, periods=periods, stamp=stamp, active=active, params=self.params)
        if reestimate and not fresh:
            columns = available(x, target=self.target)
            same = columns == self.columns
            self.columns = columns
            warm = self.params if self.config.model.warm_start and same else None
            new = self._estimate(x, periods, start=warm, day=day)
            refit = news.fit(x[:, self.columns], new, periods)
            for w in active:
                after = news.value_at(refit, series=self.target_column, row=self.row_of(w.quarter))
                self.changes.append(
                    _change(
                        stamp,
                        w.quarter,
                        "reestimation",
                        before=self.nowcast[w.quarter],
                        revisions=0.0,
                        news=0.0,
                        after=after,
                    )
                )
                self.nowcast[w.quarter] = after
            self.params = new
        tracked = {w.quarter for w in active}
        for q in tracked:
            self.nowcasts.append({"date": stamp, "quarter": str(q), "nowcast": self.nowcast[q]})
        for q in [q for q in self.nowcast if q not in tracked]:
            del self.nowcast[q]
        self.previous = x

    def _update(
        self,
        x: Array,
        *,
        periods: pd.PeriodIndex,
        stamp: pd.Timestamp,
        active: list[Window],
        params: dfm.Params,
    ) -> None:
        """Entries for quarters seen for the first time; for the others, the change since the
        previous day split into revisions and the news of each release."""
        cols = list(self.columns)
        now = x[:, cols]
        fit_new = news.fit(now, params, periods)
        old = _pad(self.previous, x.shape[0])[:, cols] if self.previous is not None else None
        fit_rev = (
            news.fit(np.where(np.isnan(old), np.nan, now), params, periods)
            if old is not None
            else None
        )
        for w in active:
            q, row = w.quarter, self.row_of(w.quarter)
            if q not in self.nowcast or old is None or fit_rev is None:
                value = news.value_at(fit_new, series=self.target_column, row=row)
                self.changes.append(
                    _change(stamp, q, "entry", before=np.nan, revisions=0.0, news=0.0, after=value)
                )
                self.nowcast[q] = value
                continue
            ch = news.decompose(
                old=self.nowcast[q],
                rev=fit_rev,
                new=fit_new,
                x_old=old,
                x_new=now,
                target=self.target_column,
                row=row,
            )
            if abs(ch.residual) > IDENTITY_TOLERANCE:
                raise RuntimeError(
                    f"{stamp:%Y-%m-%d} {q}: the news decomposition misses by {ch.residual:.2e}"
                )
            self.changes.append(
                _change(
                    stamp,
                    q,
                    "update",
                    before=ch.old,
                    revisions=ch.revisions,
                    news=ch.news,
                    after=ch.new,
                )
            )
            for r in ch.releases:
                self.releases.append(
                    {
                        "date": stamp,
                        "quarter": str(q),
                        "series": self.ids[self.columns[r.series]],
                        "period": str(periods[r.row]),
                        "actual": r.actual,
                        "forecast": r.forecast,
                        "weight": r.weight,
                        "impact": r.impact,
                    }
                )
            self.nowcast[q] = ch.new

    def _estimate(
        self, x: Array, periods: pd.PeriodIndex, *, start: dfm.Params | None, day: np.datetime64
    ) -> dfm.Params:
        began = time.monotonic()
        cols = list(self.columns)
        structure = structure_of(self.config, self.columns)
        reported = [0]

        def progress(n: int, loglik: float) -> None:
            # Accelerated EM counts several EM steps per cycle, so report on crossing.
            if n // EM_REPORT_EVERY > reported[0]:
                reported[0] = n // EM_REPORT_EVERY
                self.say(f"    EM on {day}: iteration {n:,}, log likelihood {loglik:,.2f}")

        params = dfm.estimate(
            x[:, cols],
            structure,
            periods=periods,
            settings=self.settings,
            start=start,
            progress=progress,
        )
        if not params.converged:
            self.say(f"  EM on {day} stopped at {params.iterations} iterations without converging")
        self.estimations.append(
            {
                "date": pd.Timestamp(day),
                "iterations": params.iterations,
                "converged": params.converged,
                "loglik": params.loglik,
                "seconds": time.monotonic() - began,
                "warm": start is not None,
                "scales": list(params.pandemic.scales),
                "series": [self.ids[i] for i in self.columns],
                "states": structure.n_states,
                "blocks": structure.n_blocks,
                "rho": params.pandemic.rho,
            }
        )
        return params


def run(
    config: Config,
    series: tuple[vintages.SeriesVintages, ...],
    *,
    advance: pd.DataFrame,
    gdpnow: pd.DataFrame,
    say: Say = print,
) -> History:
    spans = windows(config, advance, gdpnow)
    if not spans:
        raise ValueError("no quarter has both an advance estimate and GDPNow nowcasts")
    days, estimation_days = compute_days(series, spans)
    say(
        f"Real-time run: {len(spans)} quarters ({spans[0].quarter} to {spans[-1].quarter}), "
        f"{len(days):,} days on which the nowcast can change, {len(estimation_days)} re-estimations"
    )
    runner = _Runner(config, series, say)
    started = time.monotonic()
    step = max(1, len(days) // 40)
    for n, day in enumerate(days, start=1):
        active = [w for w in spans if w.active(day)]
        if active:
            runner.day(day, active, reestimate=day in estimation_days)
        if n % step == 0 or n == len(days):
            say(
                f"  [{time.monotonic() - started:>6.0f} s] {n:>5,} / {len(days):,} days "
                f"({pd.Timestamp(day):%Y-%m-%d})"
            )
    return History(
        nowcasts=pd.DataFrame(runner.nowcasts),
        changes=pd.DataFrame(runner.changes),
        releases=pd.DataFrame(runner.releases, columns=RELEASE_COLUMNS),
        estimations=pd.DataFrame(runner.estimations),
    )


def _change(
    stamp: pd.Timestamp,
    q: pd.Period,
    kind: str,
    *,
    before: float,
    revisions: float,
    news: float,
    after: float,
) -> dict[str, object]:
    reestimation = after - before if kind == "reestimation" else 0.0
    return {
        "date": stamp,
        "quarter": str(q),
        "kind": kind,
        "before": before,
        "revisions": revisions,
        "news": news,
        "reestimation": reestimation,
        "after": after,
    }
