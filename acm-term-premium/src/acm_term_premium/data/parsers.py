"""Parsers for the three source formats, each validated on load."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from acm_term_premium.data.source import SourceError

GSW_PARAMETERS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
# The GSW file marks parameters of the four-parameter Nelson-Siegel period this way.
_GSW_UNUSED = -999.99


def parse_gsw(text: str) -> pd.DataFrame:
    """Daily Svensson parameters indexed by date.

    The file has a free-text preamble; the table starts at the line beginning
    with `Date,`. Before 1980 the curve is Nelson-Siegel: BETA3 is zero and TAU2
    carries the unused marker, which is mapped to a zero fourth term.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Date,")), None)
    if start is None:
        raise SourceError("GSW file has no 'Date,' header line")
    frame = pd.read_csv(io.StringIO("\n".join(lines[start:])), na_values=["NA"])
    missing = [c for c in ["Date", *GSW_PARAMETERS] if c not in frame.columns]
    if missing:
        raise SourceError(f"GSW file lacks columns {missing}")
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y-%m-%d")
    params = frame.set_index("Date")[GSW_PARAMETERS].astype(float)
    params = params.dropna(subset=["BETA0", "BETA1", "BETA2", "TAU1"])
    unused = np.isclose(params["TAU2"], _GSW_UNUSED) | params["TAU2"].isna()
    unused |= np.isclose(params["BETA3"], _GSW_UNUSED) | params["BETA3"].isna()
    params.loc[unused, "BETA3"] = 0.0
    params.loc[unused, "TAU2"] = np.nan
    if (params["TAU1"] <= 0).any():
        raise SourceError("GSW TAU1 must be positive")
    if not params.index.is_monotonic_increasing or params.index.has_duplicates:
        raise SourceError("GSW dates are not strictly increasing")
    params.index.name = "date"
    return params


def svensson_yields(params: pd.DataFrame, months: np.ndarray) -> pd.DataFrame:
    """Continuously compounded zero yields, in decimals, at maturities in months."""
    m = np.asarray(months, dtype=float)[None, :] / 12.0
    b0, b1, b2, b3 = (params[c].to_numpy()[:, None] for c in ["BETA0", "BETA1", "BETA2", "BETA3"])
    t1 = params["TAU1"].to_numpy()[:, None]
    t2 = params["TAU2"].to_numpy()[:, None]
    x1 = m / t1
    f1 = (1.0 - np.exp(-x1)) / x1
    level = b0 + b1 * f1 + b2 * (f1 - np.exp(-x1))
    with np.errstate(invalid="ignore", divide="ignore"):
        x2 = m / t2
        f2 = (1.0 - np.exp(-x2)) / x2
        fourth = np.where(np.isnan(t2), 0.0, b3 * (f2 - np.exp(-x2)))
    return pd.DataFrame(
        (level + fourth) / 100.0, index=params.index, columns=[int(x) for x in months]
    )


def parse_fred(text: str, series_id: str) -> pd.Series:
    """FRED graph CSV: a date column and one value column; '.' marks missing."""
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    date_column = next(
        (c for c in frame.columns if c.lower() in {"observation_date", "date"}), None
    )
    if date_column is None or series_id not in frame.columns:
        raise SourceError(f"FRED file for {series_id} has columns {list(frame.columns)}")
    series = pd.Series(
        frame[series_id].astype(float).to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(frame[date_column]), name="date"),
        name=series_id,
    ).dropna()
    if not series.index.is_monotonic_increasing:
        raise SourceError(f"FRED {series_id} dates are not increasing")
    return series


_PUBLISHED = re.compile(r"^ACM(Y|TP|RNY)(\d{2})$")


def parse_acm_workbook(path: Path, sheet: str) -> pd.DataFrame:
    """The NY Fed workbook as a long table.

    Columns: date, maturity_years, fitted, term_premium, risk_neutral.

    Values are converted from percent to decimals to match the model output.
    """
    frame = pd.read_excel(path, sheet_name=sheet, engine="calamine")
    if "DATE" not in frame.columns:
        raise SourceError(f"sheet {sheet!r} has no DATE column")
    dates = pd.to_datetime(frame["DATE"], format="%d-%b-%Y", errors="coerce")
    if dates.isna().any():
        dates = pd.to_datetime(frame["DATE"])
    names = {"Y": "fitted", "TP": "term_premium", "RNY": "risk_neutral"}
    records: dict[tuple[str, int], pd.Series] = {}
    for column in frame.columns:
        match = _PUBLISHED.match(str(column))
        if match:
            kind, years = names[match.group(1)], int(match.group(2))
            records[(kind, years)] = frame[column].astype(float) / 100.0
    if not records:
        raise SourceError(f"sheet {sheet!r} has no ACMY/ACMTP/ACMRNY columns")
    wide = pd.DataFrame(records)
    wide.index = pd.DatetimeIndex(dates, name="date")
    long = wide.stack(level=1, future_stack=True).reset_index()
    long.columns = ["date", "maturity_years", *long.columns[2:]]
    return long.sort_values(["date", "maturity_years"]).reset_index(drop=True)
