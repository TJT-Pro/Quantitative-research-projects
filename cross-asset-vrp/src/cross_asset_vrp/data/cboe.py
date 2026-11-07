"""Cboe volatility indices and CFE VX futures: URLs, the expiry calendar and the parsers.

Index histories come in two layouts. VIX, VXN, RVX, VXEEM, VIX9D and VIX3M have
DATE,OPEN,HIGH,LOW,CLOSE; GVZ and OVX have DATE,<SYMBOL> with one value per day. Dates
are MM/DD/YYYY. The value used is the close (or the single value).

VX futures: one CSV per monthly contract. Expiries from 2013 are at
cfe_futures/VX_<YYYY-MM-DD>.csv with ISO dates; expiries to the end of 2013 are in the
archive at cfe_archive/CFE_<month code><yy>_VX.csv with MM/DD/YYYY dates. The per-contract
files of the first 2013 expiries start partway through the contract's life, so for expiries
held in both places the two files are merged, the per-contract file winning on a date both
have. Both have the header
Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest.
A settle of zero means no settlement that day. Before 26 March 2007 the contract traded on
ten times the index (the old VBI), so those settles are divided by ten in the build and
checked against VIX.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from functools import cache
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from cross_asset_vrp.config import Config

MONTH_CODES = "FGHJKMNQUVXZ"
FUTURES_COLUMNS = ["Trade Date", "Futures", "Settle"]


class SchemaError(ValueError):
    """A source file does not have the layout this pipeline was written against."""


# Index histories -----------------------------------------------------------------------------


def index_url(config: Config, symbol: str) -> str:
    return f"{config.sources.cboe_index}/{symbol}_History.csv"


def read_index(path: Path, *, symbol: str) -> pd.DataFrame:
    """date, value (the close, in volatility points)."""
    frame = pd.read_csv(path)
    columns = [c.strip().upper() for c in frame.columns]
    frame.columns = columns
    if "DATE" not in columns:
        raise SchemaError(f"{path.name}: no DATE column; found {columns}")
    value = "CLOSE" if "CLOSE" in columns else symbol.upper()
    if value not in columns:
        raise SchemaError(f"{path.name}: neither CLOSE nor {symbol} in {columns}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["DATE"], format="%m/%d/%Y", errors="coerce"),
            "value": pd.to_numeric(frame[value], errors="coerce"),
        }
    )
    out = out.dropna()
    out = out.loc[out["value"] > 0]
    return out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


# VX expiry calendar -------------------------------------------------------------------------


@cache
def _holidays(year: int) -> frozenset[date]:
    return frozenset(holidays.financial_holidays("NYSE", years=year).keys())


def is_business_day(day: date) -> bool:
    return day.weekday() < 5 and day not in _holidays(day.year)


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


def monthly_expiry(year: int, month: int) -> date:
    """CFE rule book: the Wednesday 30 days before the third Friday of the following month;
    if that Wednesday or that Friday is an exchange holiday, the business day before."""
    following = (year + month // 12, month % 12 + 1)
    friday = _third_friday(*following)
    wednesday = friday - timedelta(days=30)
    if not is_business_day(wednesday) or not is_business_day(friday):
        day = wednesday - timedelta(days=1)
        while not is_business_day(day):
            day -= timedelta(days=1)
        return day
    return wednesday


def expiries(config: Config, *, last: date) -> list[date]:
    """Monthly expiries from the configured first one to `last`."""
    start = pd.Period(config.futures.first_expiry, freq="M")
    end = pd.Period(last, freq="M")
    return [monthly_expiry(p.year, p.month) for p in pd.period_range(start, end, freq="M")]


def gaps_inside_listing(expected: list[date], present: set[date], *, run: int = 12) -> list[date]:
    """Expiries missing between the start of the monthly cycle and the last listed contract.

    VX was not listed every month at first, and contracts far ahead are not listed yet, so
    those months are missing by design. Once `run` consecutive monthly contracts exist, the
    cycle is monthly: a gap after that, before the last listed contract, means a download that
    failed, not a contract that never existed."""
    ordered = sorted(expected)
    flags = [e in present for e in ordered]
    start = next((i for i in range(len(ordered) - run + 1) if all(flags[i : i + run])), None)
    if start is None or not any(flags):
        return []
    last = max(i for i, ok in enumerate(flags) if ok)
    return [ordered[i] for i in range(start, last) if not flags[i]]


def in_new_files(config: Config, expiry: date) -> bool:
    return pd.Period(expiry, freq="M") >= pd.Period(config.futures.new_files_from, freq="M")


def in_archive(config: Config, expiry: date) -> bool:
    return pd.Period(expiry, freq="M") <= pd.Period(config.futures.archive_until, freq="M")


def archive_name(expiry: date) -> str:
    return f"CFE_{MONTH_CODES[expiry.month - 1]}{expiry.year % 100:02d}_VX.csv"


def archive_url(config: Config, expiry: date) -> str:
    return f"{config.sources.cfe_archive}/{archive_name(expiry)}"


def futures_url(config: Config, expiry: date) -> str:
    """The contract's primary file: the per-contract file when one exists, else the archive."""
    if in_new_files(config, expiry):
        return f"{config.sources.cfe_futures}/VX_{expiry.isoformat()}.csv"
    return archive_url(config, expiry)


def needs_archive_too(config: Config, expiry: date) -> bool:
    """Expiries held both in the per-contract files and in the archive."""
    return in_new_files(config, expiry) and in_archive(config, expiry)


def merge_sources(primary: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """One contract from both files; on a date both have, the per-contract file wins."""
    out = pd.concat([archive, primary], ignore_index=True)
    return (
        out.drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def gaps(
    trade_dates: pd.Series, *, longer_than: int
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Every stretch between consecutive dates longer than `longer_than` calendar days, as
    (last date before, first date after, days)."""
    d = np.sort(pd.DatetimeIndex(trade_dates.unique()).to_numpy(dtype="datetime64[D]"))
    step = np.diff(d).astype(np.int64)
    return [
        (pd.Timestamp(d[i]), pd.Timestamp(d[i + 1]), int(step[i]))
        for i in np.flatnonzero(step > longer_than)
    ]


def longest_gap(trade_dates: pd.Series) -> tuple[int, pd.Timestamp | None]:
    """The longest stretch in calendar days between consecutive dates, and the date it ends."""
    d = np.sort(pd.DatetimeIndex(trade_dates.unique()).to_numpy(dtype="datetime64[D]"))
    if d.size < 2:
        return 0, None
    step = np.diff(d).astype(np.int64)
    at = int(np.argmax(step))
    return int(step[at]), pd.Timestamp(d[at + 1])


def _futures_frame(lines: list[str], *, name: str) -> pd.DataFrame:
    """The table under the header. Rows in the 2004-2013 archive files often end in a trailing
    comma, one more field than the header has; empty fields past the header are dropped, and a
    row with a value past the header is refused."""
    rows = list(csv.reader(lines))
    header = [h.strip() for h in rows[0]]
    width = len(header)
    body = []
    for number, row in enumerate(rows[1:], start=2):
        if not any(field.strip() for field in row):
            continue
        extra = row[width:]
        if any(field.strip() for field in extra):
            raise SchemaError(
                f"{name}: line {number} of the table has values past the header: {row}"
            )
        body.append(row[:width] + [""] * (width - len(row[:width])))
    return pd.DataFrame(body, columns=header)


def read_futures(path: Path, *, expiry: date) -> pd.DataFrame:
    """trade_date, expiry, settle (as published; the build rescales the pre-2007 contracts)."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Trade Date")), None)
    if start is None:
        raise SchemaError(f"{path.name}: no 'Trade Date' header")
    frame = _futures_frame(lines[start:], name=path.name)
    missing = [c for c in FUTURES_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"{path.name}: missing {missing}; found {list(frame.columns)}")
    raw = frame["Trade Date"].astype(str).str.strip()
    iso = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    us = pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce")
    out = pd.DataFrame(
        {
            "trade_date": iso.fillna(us),
            "expiry": pd.Timestamp(expiry),
            "settle": pd.to_numeric(frame["Settle"], errors="coerce"),
        }
    )
    out = out.dropna()
    out = out.loc[(out["settle"] > 0) & (out["trade_date"] <= out["expiry"])]
    return (
        out.drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
