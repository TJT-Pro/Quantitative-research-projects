"""Synthetic sources in the exact upstream formats, for offline runs, CI and the e2e test.

Planted structure, so the pipeline's answers can be checked:
- The transformed panel is drawn from a factor model with the layout the model estimates
  (models.simulate), with the shocks of March to May 2020 scaled up and decaying after, and
  turned back into levels through the inverse of each series' transformation.
- Each value is first published on a realistic lag after its period ends (payrolls on the
  first Friday, the Philadelphia survey before its month is over, GDP four weeks after the
  quarter) and revised once a month later, so ALFRED rows carry real-time intervals.
- GDPNow nowcasts are the advance estimate plus noise that shrinks toward the release.
- The 2-year yield moves on release days by a planted multiple of the surprise in the
  released data, so the market test has a known positive answer.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

from realtime_nowcast.config import Config, SeriesConfig
from realtime_nowcast.data import build, fred, vintages
from realtime_nowcast.data.download import mirror_path
from realtime_nowcast.evaluation.realtime import structure_of
from realtime_nowcast.models import simulate
from realtime_nowcast.models.dfm import Pandemic

# ALFRED's history of these series starts later than the data.
FIRST_VINTAGE = pd.Timestamp("1995-01-03")
# ALFRED's real record of the Philadelphia survey starts on 2015-04-16; the fixture starts it
# inside the CI window, so a series entering the model late is exercised by the e2e test.
LATE_RECORD = {"GACDFSA066MSFRBPHI": pd.Timestamp("2012-04-16")}
REVISION_DAYS = 30
REVISION_SD = 0.15  # in standard deviations of the transformed series
PANDEMIC = {"2020-03": 8.0, "2020-04": 5.0, "2020-05": 3.0}
PANDEMIC_DECAY = 0.6
YIELD_BETA = 6.0  # basis points per standard deviation of surprise, summed over a day's releases
YIELD_NOISE = 4.0
GDPNOW_SD_START, GDPNOW_SD_END = 1.6, 0.6
GDPNOW_FIRST_DAY = 28  # GDPNow starts about four weeks into the quarter

# Days after the end of the period (negative: before it ends), then moved off weekends.
LAG_DAYS = {
    "JTSJOL": 38,
    "CPIAUCSL": 13,
    "DGORDER": 26,
    "RSAFS": 15,
    "HOUST": 17,
    "INDPRO": 16,
    "DSPIC96": 29,
    "BOPTEXP": 35,
    "BOPTIMP": 35,
    "TTLCONS": 31,
    "IR": 13,
    "CPILFESL": 13,
    "PCEPILFE": 29,
    "PCEPI": 29,
    "PERMIT": 17,
    "TCU": 16,
    "BUSINV": 44,
    "IQ": 13,
    "GACDISA066MSFRBNY": -15,
    "PCEC96": 29,
    "GACDFSA066MSFRBPHI": -10,
    "GDPC1": 28,
    "ULCNFB": 36,
}
FIRST_FRIDAY = {"PAYEMS", "UNRATE"}
STARTS_LATE = {"JTSJOL": "2000-12"}
# Location and scale of each transformed series, so the levels look like the real ones.
SCALE = {"chg": (0.0, 1.0), "pch": (0.25, 0.6), "pca": (2.5, 2.5), "lin": (5.0, 10.0)}
LEVEL_BASE = {"chg": 1000.0, "pch": 100.0, "pca": 10000.0}
CHG_SCALE = {
    "PAYEMS": (150.0, 200.0),
    "UNRATE": (0.0, 0.15),
    "JTSJOL": (20.0, 150.0),
    "PERMIT": (2.0, 60.0),
    "TCU": (0.0, 0.5),
}


@dataclass(frozen=True)
class Planted:
    yield_beta: float = YIELD_BETA
    pandemic: tuple[float, ...] = tuple(PANDEMIC.values())


def _business(day: pd.Timestamp) -> pd.Timestamp:
    while day.weekday() >= 5:
        day += pd.Timedelta(days=1)
    return day


def _first_friday_after(month_end: pd.Timestamp) -> pd.Timestamp:
    day = month_end + pd.Timedelta(days=1)
    while day.weekday() != 4:
        day += pd.Timedelta(days=1)
    return day


def release_day(spec: SeriesConfig, period: pd.Period) -> pd.Timestamp:
    end = period.end_time.normalize()
    if spec.id in FIRST_FRIDAY:
        return _first_friday_after(end)
    return _business(end + pd.Timedelta(days=LAG_DAYS[spec.id]))


def _levels(spec: SeriesConfig, x: np.ndarray) -> np.ndarray:
    """Invert the transformation; x holds the transformed values on the grid."""
    if spec.transform == "lin":
        return x
    step = 1 if spec.frequency == "m" else 3
    out = np.full_like(x, np.nan)
    rows = np.arange(step - 1, x.size, step)
    level = LEVEL_BASE[spec.transform]
    for k, r in enumerate(rows):
        if k > 0 and not np.isnan(x[r]):
            if spec.transform == "chg":
                level = level + x[r]
            elif spec.transform == "pch":
                level = level * (1 + x[r] / 100)
            else:
                level = level * (1 + x[r] / 100) ** (step / 12)
        out[r] = level
    return out


@dataclass
class Mirror:
    root: Path

    def put(self, url: str, data: bytes) -> None:
        path = mirror_path(self.root, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _page(rows: list[dict[str, str]], *, offset: int, limit: int) -> bytes:
    body = {
        "realtime_start": fred.REALTIME_ALL[0],
        "realtime_end": fred.REALTIME_ALL[1],
        "output_type": 1,
        "file_type": "json",
        "order_by": "observation_date",
        "sort_order": "asc",
        "count": len(rows),
        "offset": offset,
        "limit": limit,
        "observations": rows[offset : offset + limit],
    }
    return json.dumps(body).encode()


def _put_pages(
    mirror: Mirror, rows: list[dict[str, str]], *, url_for: Callable[[int], str], limit: int
) -> None:
    offset = 0
    while True:
        mirror.put(url_for(offset), _page(rows, offset=offset, limit=limit))
        offset += limit
        if offset >= len(rows):
            return


def _fmt(day: pd.Timestamp) -> str:
    return day.strftime("%Y-%m-%d")


def _alfred_rows(
    spec: SeriesConfig,
    periods: pd.PeriodIndex,
    *,
    levels: np.ndarray,
    revised: np.ndarray,
    today: pd.Timestamp,
) -> list[dict[str, str]]:
    rows = []
    for p, v1, v2 in zip(periods, levels, revised, strict=True):
        if np.isnan(v1):
            continue
        period = p.asfreq("Q") if spec.frequency == "q" else p
        first = max(release_day(spec, period), LATE_RECORD.get(spec.id, FIRST_VINTAGE))
        second = first + pd.Timedelta(days=REVISION_DAYS)
        if first > today:
            continue
        date = (period.start_time if spec.frequency == "q" else p.start_time).strftime("%Y-%m-%d")
        if second <= today:
            rows.append(
                {
                    "realtime_start": _fmt(first),
                    "realtime_end": _fmt(second - pd.Timedelta(days=1)),
                    "date": date,
                    "value": f"{v1:.6f}",
                }
            )
            rows.append(
                {
                    "realtime_start": _fmt(second),
                    "realtime_end": "9999-12-31",
                    "date": date,
                    "value": f"{v2:.6f}",
                }
            )
        else:
            rows.append(
                {
                    "realtime_start": _fmt(first),
                    "realtime_end": "9999-12-31",
                    "date": date,
                    "value": f"{v2:.6f}",
                }
            )
    rows.sort(key=lambda r: (r["date"], r["realtime_start"]))
    return rows


ARCHIVE_HEADER = [
    "Forecast Date",
    "Quarter being forecasted",
    "PCE",
    "PCE Goods",
    "PCE Services",
    "GPDI",
    "Fixed Investment",
    "Business Fixed Investment",
    "Equipment",
    "Intellectual Property Products",
    "Structures",
    "Residential",
    "Government",
    "Federal Govt",
    "S&L",
    "Imports",
    "Goods imports",
    "Services imports",
    "Exports",
    "Goods exports",
    "Services exports",
    "Net Exports (current, $Bil 2009)",
    "Net Exports (previous, $Bil 2009)",
    "Change in net exports ($Bil 2009)",
    "Previous change in private inventories ($Bil 2009)",
    "Current change in private inventories ($Bil 2009)",
    "Change in inventory investment ($Bil 2009)",
    "GDP Nowcast",
    "Final Sales",
    "Final Sales to Domestic Purchasers",
    "Final Sales to Private Domestic Purchasers",
    "Advance Estimate From BEA",
    "Publication Date of Advance Estimate",
    "Days until advance estimate",
    "Forecast Error",
    "Data releases",
]
DEEP_HEADER = [
    *ARCHIVE_HEADER[:21],
    "Current Quarter Net Exports (real, $Bil)",
    "Previous Quarter Net Exports (real, $Bil)",
    "Change in net exports (real, $Bil)",
    "Previous change in private inventories (real, $Bil)",
    "Current change in private inventories (real, $Bil)",
    "Change in inventory investment (real, $Bil)",
    "GDP Nowcast",
    "Final Sales",
    "Advance Estimate From BEA",
    "Publication Date of Advance Estimate",
    "Days until advance estimate",
    "Forecast Error",
]
RECORD_HEADER = [
    "Quarter being forecasted",
    "Model Forecast Right Before BEA's Advance Estimate",
    "BEA's Advance Estimate",
    "Release Date",
    None,
    "Error",
    "Absolute Error",
    "Squared Error",
]
DEEP_LAST = pd.Period("2014Q1", freq="Q")


def _workbook(nowcasts: pd.DataFrame, advance: pd.DataFrame) -> bytes:
    book = Workbook()
    book.properties.created = datetime(2026, 1, 1)
    book.properties.modified = datetime(2026, 1, 1)
    readme = book.create_sheet("ReadMe", 0)
    book.remove(book.worksheets[1])
    readme.append(["Synthetic GDPNow workbook written by realtime_nowcast.data.fixture"])
    deep = book.create_sheet("TrackingDeepArchives")
    deep.append(DEEP_HEADER)
    arch = book.create_sheet("TrackingArchives")
    arch.append([*ARCHIVE_HEADER, None, None])
    for quarter, day, value in zip(
        nowcasts["quarter"].astype(str),
        pd.to_datetime(nowcasts["date"]),
        nowcasts["nowcast"].to_numpy(dtype=float),
        strict=True,
    ):
        q = pd.Period(quarter, freq="Q")
        sheet, header = (deep, DEEP_HEADER) if q <= DEEP_LAST else (arch, ARCHIVE_HEADER)
        cells: list[object] = ["#N/A"] * len(header)
        cells[header.index("Forecast Date")] = day.to_pydatetime()
        cells[header.index("Quarter being forecasted")] = q.end_time.normalize().to_pydatetime()
        cells[header.index("GDP Nowcast")] = float(value)
        sheet.append(cells)
    record = book.create_sheet("TrackRecord")
    record.append(RECORD_HEADER)
    ordered = advance.sort_values("quarter", ascending=False)
    for quarter, first, release in zip(
        ordered["quarter"].astype(str),
        ordered["advance"].to_numpy(dtype=float),
        pd.to_datetime(ordered["release"]),
        strict=True,
    ):
        last = nowcasts.loc[nowcasts["quarter"] == quarter, "nowcast"]
        if last.empty:
            continue
        q = pd.Period(quarter, freq="Q")
        err = float(first) - float(last.iloc[-1])
        record.append(
            [
                q.end_time.normalize().to_pydatetime(),
                float(last.iloc[-1]),
                float(first),
                release.to_pydatetime(),
                None,
                err,
                abs(err),
                err**2,
            ]
        )
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def _revised(spec: SeriesConfig, levels: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Levels after one revision of `noise` in the units of the transformed series."""
    if spec.transform in ("chg", "lin"):
        return np.asarray(levels + noise, dtype=np.float64)
    # A level revision of r percent moves pch growth by r and annualized pca growth by about
    # 4r, so the noise (in growth units) is converted before it is applied.
    per_level = 1.0 if spec.transform == "pch" else 4.0
    return np.asarray(levels * (1 + noise / per_level / 100), dtype=np.float64)


def write_fixture(
    *, config: Config, mirror: Path, seed: int = 7, end: str = "2022-06-30"
) -> Planted:
    """Write every source under `mirror`. `end` is the last day any value is published."""
    rng = np.random.default_rng(seed)
    m = Mirror(root=mirror)
    specs = config.panel.ordered
    structure = structure_of(config)
    today = pd.Timestamp(end)
    start = pd.Period(config.panel.observation_start, freq="M")
    periods = pd.period_range(start, today.to_period("M") + 3, freq="M")
    if start.month != 1:
        raise ValueError("the fixture's grid starts in January so third months close quarters")
    truth = simulate.truth(structure, rng)
    free = tuple(pd.Period(month, freq="M") for month in PANDEMIC)
    scale = Pandemic(free=free, scales=tuple(PANDEMIC.values()), rho=PANDEMIC_DECAY).scale(periods)
    y, states = simulate.draw(structure, truth, periods=len(periods), rng=rng, scale=scale)
    surprises: dict[pd.Timestamp, float] = {}
    target_rows: list[dict[str, str]] = []
    for j, spec in enumerate(specs):
        loc, sd = CHG_SCALE.get(spec.id, SCALE[spec.transform])
        x = loc + sd * y[:, j]
        if spec.id in STARTS_LATE:
            x[periods < pd.Period(STARTS_LATE[spec.id], freq="M")] = np.nan
        lv = _levels(spec, x)
        if spec.id in STARTS_LATE:
            lv[periods < pd.Period(STARTS_LATE[spec.id], freq="M")] = np.nan
        revised = _revised(spec, lv, REVISION_SD * sd * rng.standard_normal(lv.size))
        rows = _alfred_rows(spec, periods, levels=lv, revised=revised, today=today)
        if spec.id == config.panel.target:
            target_rows = rows
        _put_pages(
            m,
            rows,
            url_for=fred.vintages_pages(config, spec.id),
            limit=config.download.page_limit,
        )
        if spec.frequency == "m":
            # Surprise of each first release: the datum minus its expectation given the
            # factors of the month before, in standard deviations.
            pred = (truth.c[j] @ (truth.a @ states[:-1].T)).T
            surprise = y[1:, j] - pred
            for p, s in zip(periods[1:], surprise, strict=True):
                if np.isnan(s):
                    continue
                day = release_day(spec, p)
                if FIRST_VINTAGE <= day <= today:
                    sign = 1.0 if truth.c[j, 0] >= 0 else -1.0
                    surprises[day] = surprises.get(day, 0.0) + sign * float(s)
    _yields(config, m, surprises, rng=rng, today=today)
    target = next(s for s in specs if s.id == config.panel.target)
    advance = build.advance_estimates(vintages.from_rows(target, pd.DataFrame(target_rows)))
    nowcasts = _gdpnow_nowcasts(config, advance, rng=rng)
    m.put(config.sources.gdpnow[0], _workbook(nowcasts, advance))
    return Planted()


def _gdpnow_nowcasts(
    config: Config, advance: pd.DataFrame, *, rng: np.random.Generator
) -> pd.DataFrame:
    first = pd.Period(config.evaluation.first_quarter, freq="Q")
    rows = []
    for quarter, release, value in zip(
        advance["quarter"].astype(str),
        pd.to_datetime(advance["release"]),
        advance["advance"].to_numpy(dtype=float),
        strict=True,
    ):
        q = pd.Period(quarter, freq="Q")
        if q < first:
            continue
        span = pd.bdate_range(
            q.start_time + pd.Timedelta(days=GDPNOW_FIRST_DAY), release - pd.Timedelta(days=1)
        )
        days = pd.DatetimeIndex(sorted({*span[::3], span[-1]}))
        sd = np.linspace(GDPNOW_SD_START, GDPNOW_SD_END, len(days))
        noise = np.cumsum(rng.standard_normal(len(days)))[::-1]
        noise = noise / np.sqrt(np.arange(len(days), 0, -1))
        for d, s, e in zip(days, sd, noise, strict=True):
            rows.append({"date": d, "quarter": str(q), "nowcast": float(value + s * e)})
    return pd.DataFrame(rows)


def _yields(
    config: Config,
    m: Mirror,
    surprises: dict[pd.Timestamp, float],
    *,
    rng: np.random.Generator,
    today: pd.Timestamp,
) -> None:
    start = pd.Period(config.evaluation.first_quarter, freq="Q").start_time - pd.DateOffset(years=1)
    days = pd.bdate_range(start, today)
    by_day = pd.Series(surprises, dtype=float).reindex(days, fill_value=0.0)
    for k, series in enumerate(config.market.yields):
        beta = YIELD_BETA if series == config.market.primary else YIELD_BETA / 2
        level = 2.0 + k
        rows = []
        for d, s in zip(days, by_day.to_numpy(), strict=True):
            level += float(beta * s + YIELD_NOISE * rng.standard_normal()) / 100.0
            rows.append(
                {
                    "realtime_start": _fmt(today),
                    "realtime_end": _fmt(today),
                    "date": _fmt(d),
                    "value": f"{level:.2f}",
                }
            )
        _put_pages(
            m,
            rows,
            url_for=fred.current_pages(config, series, start=start.strftime("%Y-%m-%d")),
            limit=config.download.page_limit,
        )
