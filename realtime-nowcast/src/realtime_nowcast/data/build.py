"""Parse data/raw into the tables every later stage reads.

- vintages: series, period (month index), start, end, value: every published value and the
  days it was current, one row per ALFRED row;
- advance: quarter, release, advance: the first published growth of real GDP for each
  quarter (annualized, from the levels of that quarter and the one before in the vintage of
  its release day), which is what both nowcasts aim at;
- gdpnow: date, quarter, nowcast; track_record: the Atlanta Fed's own advance estimates;
- yields: date, series, value (percent).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from realtime_nowcast.config import Config
from realtime_nowcast.data import gdpnow, vintages
from realtime_nowcast.data.fetch import ALFRED_PREFIX, GDPNOW_KEY, YIELDS_PREFIX
from realtime_nowcast.data.manifest import Manifest
from realtime_nowcast.paths import ProjectPaths

Say = Callable[[str], object]


@dataclass(frozen=True)
class Tables:
    vintages: pd.DataFrame
    advance: pd.DataFrame
    gdpnow: pd.DataFrame
    track_record: pd.DataFrame
    yields: pd.DataFrame

    def names(self) -> dict[str, pd.DataFrame]:
        return {
            "vintages": self.vintages,
            "advance": self.advance,
            "gdpnow": self.gdpnow,
            "track_record": self.track_record,
            "yields": self.yields,
        }


def _rows(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(
        payload["observations"], columns=["realtime_start", "realtime_end", "date", "value"]
    )


def series_vintages(config: Config, frame: pd.DataFrame) -> tuple[vintages.SeriesVintages, ...]:
    """The vintages table back as one SeriesVintages per series, in model order."""
    out = []
    for s in config.panel.ordered:
        part = frame.loc[frame["series"] == s.id].sort_values(["period", "start"])
        out.append(
            vintages.SeriesVintages(
                id=s.id,
                frequency=s.frequency,
                period=part["period"].to_numpy(dtype=np.int64),
                start=part["start"].to_numpy().astype("datetime64[D]"),
                end=part["end"].to_numpy().astype("datetime64[D]"),
                value=part["value"].to_numpy(dtype=np.float64),
            )
        )
    return tuple(out)


def advance_estimates(gdp: vintages.SeriesVintages) -> pd.DataFrame:
    """quarter, release, advance: growth of each quarter in the vintage of its first release."""
    first = vintages.first_release(gdp)
    rows = []
    for period, release in zip(first["period"], first["release"], strict=True):
        day = np.datetime64(release, "D")
        live = (gdp.start <= day) & (gdp.end >= day)
        now = gdp.value[live & (gdp.period == period)]
        before = gdp.value[live & (gdp.period == period - 3)]
        if now.size != 1 or before.size != 1 or np.isnan(before[0]):
            continue
        quarter = vintages.index_month(int(period)).asfreq("Q")
        rows.append(
            {
                "quarter": str(quarter),
                "release": pd.Timestamp(day),
                "advance": float(100.0 * ((now[0] / before[0]) ** 4 - 1.0)),
            }
        )
    return pd.DataFrame(rows, columns=["quarter", "release", "advance"])


def build_tables(*, config: Config, paths: ProjectPaths, say: Say = print) -> Tables:
    manifest = Manifest.read(paths.manifest)
    if not manifest.entries:
        raise FileNotFoundError("data/manifest.json is empty: run `rtn fetch` first")
    data = paths.data
    frames = []
    for s in config.panel.ordered:
        key = f"{ALFRED_PREFIX}{s.id}"
        if key not in manifest.entries:
            raise FileNotFoundError(f"{s.id} is not downloaded: run `rtn fetch`")
        v = vintages.from_rows(s, _rows(data / manifest.entries[key].path))
        frames.append(
            pd.DataFrame(
                {
                    "series": s.id,
                    "period": v.period,
                    "start": v.start,
                    "end": v.end,
                    "value": v.value,
                }
            )
        )
        say(f"  {s.id:<20} {len(v.period):>8,} values")
    table = pd.concat(frames, ignore_index=True)
    gdp = next(v for v in series_vintages(config, table) if v.id == config.panel.target)
    advance = advance_estimates(gdp)
    workbook = data / manifest.entries[GDPNOW_KEY].path
    now = gdpnow.read_nowcasts(workbook).assign(quarter=lambda f: f["quarter"].astype(str))
    record = gdpnow.read_track_record(workbook).assign(quarter=lambda f: f["quarter"].astype(str))
    yield_frames = []
    for name in config.market.yields:
        rows = _rows(data / manifest.entries[f"{YIELDS_PREFIX}{name}"].path)
        yield_frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(rows["date"]),
                    "series": name,
                    "value": pd.to_numeric(rows["value"].replace(".", np.nan)),
                }
            ).dropna()
        )
    say(
        f"{len(advance)} advance estimates, "
        f"{len(now):,} GDPNow nowcasts of {now['quarter'].nunique()} quarters"
    )
    return Tables(
        vintages=table,
        advance=advance,
        gdpnow=now,
        track_record=record,
        yields=pd.concat(yield_frames, ignore_index=True),
    )


def write_tables(tables: Tables, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.names().items():
        frame.to_parquet(folder / f"{name}.parquet", index=False)


def read_tables(folder: Path) -> Tables:
    def read(name: str) -> pd.DataFrame:
        path = folder / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: run `rtn build`")
        return pd.read_parquet(path)

    return Tables(
        vintages=read("vintages"),
        advance=read("advance"),
        gdpnow=read("gdpnow"),
        track_record=read("track_record"),
        yields=read("yields"),
    )
