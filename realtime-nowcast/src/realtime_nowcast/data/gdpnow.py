"""The Atlanta Fed's GDPNow workbook (GDPTrackingModelDataAndForecasts.xlsx).

Two sheets hold every published nowcast of completed quarters:

- TrackingDeepArchives: 2011:Q3 to 2014:Q1;
- TrackingArchives: 2014:Q2 onward.

Both have a header row with "Forecast Date", "Quarter being forecasted" (the last day of the
quarter) and "GDP Nowcast" (annualized percent growth), among about thirty other columns
whose order differs between the sheets, so columns are found by name. Cells the model did
not fill hold the text "#N/A". TrackRecord lists, per quarter, the last nowcast before the
advance estimate, the advance estimate and its release date.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook

ARCHIVES = ("TrackingDeepArchives", "TrackingArchives")
TRACK_RECORD = "TrackRecord"
DATE, QUARTER, NOWCAST = "Forecast Date", "Quarter being forecasted", "GDP Nowcast"
RECORD_QUARTER = "Quarter being forecasted"
RECORD_ADVANCE = "BEA's Advance Estimate"
RECORD_RELEASE = "Release Date"


class SchemaError(ValueError):
    """The workbook does not have the layout this pipeline was written against."""


def _day(value: Any) -> pd.Timestamp | None:
    if isinstance(value, datetime | date):
        return pd.Timestamp(value).normalize()
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _rows(path: Path, sheet: str) -> list[tuple[Any, ...]]:
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet not in book.sheetnames:
            raise SchemaError(f"{path.name}: no sheet {sheet!r}; found {book.sheetnames}")
        return list(book[sheet].iter_rows(values_only=True))
    finally:
        book.close()


def _columns(header: tuple[Any, ...], wanted: tuple[str, ...], sheet: str) -> list[int]:
    names = [str(h).strip() if h is not None else "" for h in header]
    missing = [w for w in wanted if w not in names]
    if missing:
        raise SchemaError(f"{sheet}: missing columns {missing}; found {[n for n in names if n]}")
    return [names.index(w) for w in wanted]


def read_nowcasts(path: Path) -> pd.DataFrame:
    """date, quarter (Period Q), nowcast: one row per published nowcast."""
    frames = []
    for sheet in ARCHIVES:
        rows = _rows(path, sheet)
        if not rows:
            raise SchemaError(f"{sheet} is empty")
        i_date, i_quarter, i_now = _columns(rows[0], (DATE, QUARTER, NOWCAST), sheet)
        out = []
        for row in rows[1:]:
            day, quarter, value = _day(row[i_date]), _day(row[i_quarter]), _number(row[i_now])
            if day is None or quarter is None or value is None:
                continue
            out.append({"date": day, "quarter": pd.Period(quarter, freq="Q"), "nowcast": value})
        frames.append(pd.DataFrame(out))
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise SchemaError("no nowcasts in the GDPNow archives")
    return frame.drop_duplicates(["quarter", "date"], keep="last").sort_values(
        ["quarter", "date"], ignore_index=True
    )


def read_track_record(path: Path) -> pd.DataFrame:
    """quarter, advance, release: the Atlanta Fed's own record of the advance estimates."""
    rows = _rows(path, TRACK_RECORD)
    header_at = next(
        (i for i, r in enumerate(rows) if r and RECORD_ADVANCE in [str(c).strip() for c in r]), None
    )
    if header_at is None:
        raise SchemaError(f"{TRACK_RECORD}: no header with {RECORD_ADVANCE!r}")
    cols = _columns(rows[header_at], (RECORD_QUARTER, RECORD_ADVANCE, RECORD_RELEASE), TRACK_RECORD)
    out = []
    for row in rows[header_at + 1 :]:
        quarter, advance, release = _day(row[cols[0]]), _number(row[cols[1]]), _day(row[cols[2]])
        if quarter is None or advance is None:
            continue
        out.append(
            {"quarter": pd.Period(quarter, freq="Q"), "advance": advance, "release": release}
        )
    return pd.DataFrame(out).sort_values("quarter", ignore_index=True)
