"""Readers for each source format. Every reader checks its columns and fails loudly.

Months are stored as the first day of the month (datetime64), returns in percent.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """A source file does not have the layout this pipeline was written against."""


def _require(frame: pd.DataFrame, columns: list[str], *, source: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SchemaError(f"{source}: missing columns {missing}; found {list(frame.columns)}")


def to_month(values: pd.Series) -> pd.Series:
    """Datetimes, yyyymm integers or date strings to the first day of their month."""
    if pd.api.types.is_datetime64_any_dtype(values):
        stamps = pd.to_datetime(values)
    elif pd.api.types.is_integer_dtype(values) or pd.api.types.is_float_dtype(values):
        digits = values.astype("Int64").astype(str)
        layout = "%Y%m%d" if digits.str.len().max() == 8 else "%Y%m"
        stamps = pd.to_datetime(digits, format=layout)
    else:
        text = values.astype(str).str.strip()
        stamps = pd.to_datetime(text.str.slice(0, 10), format="mixed")
    months: pd.Series = pd.Series(stamps, index=values.index)
    first_days: pd.Series = months.dt.to_period("M").dt.to_timestamp()
    return first_days


# Open Source Asset Pricing -----------------------------------------------------------------

SIGNAL_DOC_COLUMNS = {
    "Acronym": "signal",
    "Cat.Signal": "kind",
    "Predictability in OP": "predictability",
    "Signal Rep Quality": "rep_quality",
    "Authors": "authors",
    "Year": "year",
    "Journal": "journal",
    "Cat.Economic": "category",
    "SampleStartYear": "sample_start",
    "SampleEndYear": "sample_end",
    "Sign": "sign",
    "T-Stat": "paper_t",
}


def read_signal_doc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, na_values=["NA", ""], keep_default_na=False, low_memory=False)
    _require(frame, list(SIGNAL_DOC_COLUMNS), source=path.name)
    out = frame[list(SIGNAL_DOC_COLUMNS)].rename(columns=SIGNAL_DOC_COLUMNS)
    for column in ("year", "sample_start", "sample_end", "sign", "paper_t"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["signal"] = out["signal"].astype(str).str.strip()
    out["kind"] = out["kind"].astype(str).str.strip()
    return out.drop_duplicates("signal").reset_index(drop=True)


def read_long_short(path: Path, *, port: str) -> pd.DataFrame:
    """The long-short rows of an OSAP portfolio file (long layout: signalname, port, date, ret)."""
    columns = ["signalname", "port", "date", "ret"]
    try:
        frame = pd.read_parquet(path, columns=columns, filters=[("port", "==", port)])
    except (KeyError, ValueError) as error:
        raise SchemaError(f"{path.name}: cannot read {columns}: {error}") from error
    _require(frame, columns, source=path.name)
    out = pd.DataFrame(
        {
            "signal": frame["signalname"].astype(str),
            "month": to_month(frame["date"]),
            "ret": pd.to_numeric(frame["ret"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["ret"])
    if out.duplicated(["signal", "month"]).any():
        raise SchemaError(f"{path.name}: more than one {port} return per signal and month")
    return out.sort_values(["signal", "month"]).reset_index(drop=True)


# Global Factor Data ------------------------------------------------------------------------


def _first_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = sorted(n for n in archive.namelist() if n.lower().endswith(".csv"))
        if not members:
            raise SchemaError(f"{path.name}: no CSV inside the zip")
        with archive.open(members[0]) as handle:
            data = handle.read()
    return pd.read_csv(io.BytesIO(data), na_values=["na", "NA", ""], keep_default_na=False)


def _date_column(frame: pd.DataFrame, *, source: str) -> str:
    for candidate in ("date", "eom"):
        if candidate in frame.columns:
            return candidate
    raise SchemaError(f"{source}: no date or eom column; found {list(frame.columns)}")


def read_jkp_factors(path: Path) -> pd.DataFrame:
    """A JKP factor file. Country files carry n_stocks_min, the smaller leg's stock count.
    Regional aggregates carry n_countries instead: JKP build them only from country factors
    that already pass their stock minimum, so `aggregate` marks rows that need no stock filter."""
    frame = _first_csv(path)
    _require(frame, ["name", "ret"], source=path.name)
    date = _date_column(frame, source=path.name)
    aggregate = "n_stocks_min" not in frame.columns
    if aggregate and "n_countries" not in frame.columns:
        raise SchemaError(
            f"{path.name}: needs n_stocks_min (country) or n_countries (region); "
            f"found {list(frame.columns)}"
        )
    counts = frame["n_countries"] if aggregate else frame["n_stocks_min"]
    out = pd.DataFrame(
        {
            "factor": frame["name"].astype(str),
            "month": to_month(frame[date]),
            "ret": pd.to_numeric(frame["ret"], errors="coerce") * 100.0,
            "n_stocks_min": np.nan if aggregate else pd.to_numeric(counts, errors="coerce"),
            "n_countries": pd.to_numeric(counts, errors="coerce") if aggregate else np.nan,
            "aggregate": aggregate,
        }
    )
    return out.dropna(subset=["ret"]).reset_index(drop=True)


def read_jkp_market(path: Path) -> pd.DataFrame:
    frame = _first_csv(path)
    date = _date_column(frame, source=path.name)
    value = next((c for c in ("mkt_vw_exc", "ret") if c in frame.columns), None)
    if value is None:
        raise SchemaError(f"{path.name}: need ret or mkt_vw_exc; found {list(frame.columns)}")
    out = pd.DataFrame(
        {
            "month": to_month(frame[date]),
            "mkt": pd.to_numeric(frame[value], errors="coerce") * 100.0,
        }
    )
    return out.dropna(subset=["mkt"]).drop_duplicates(["month"]).reset_index(drop=True)


def read_cluster_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require(frame, ["characteristic", "cluster"], source=path.name)
    return frame.rename(columns={"characteristic": "factor"})[["factor", "cluster"]]


def read_factor_details(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=0)
    _require(frame, ["abr_jkp", "direction", "significance", "cite"], source=path.name)
    out = frame.loc[frame["abr_jkp"].notna(), ["abr_jkp", "direction", "significance", "cite"]]
    out = out.rename(columns={"abr_jkp": "factor"})
    out["factor"] = out["factor"].astype(str)
    out["significance"] = pd.to_numeric(out["significance"], errors="coerce").fillna(0) == 1
    out["direction"] = pd.to_numeric(out["direction"], errors="coerce")
    return out.drop_duplicates("factor").reset_index(drop=True)


def read_country_classification(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=0)
    _require(frame, ["excntry", "msci_development", "region"], source=path.name)
    return pd.DataFrame(
        {
            "location": frame["excntry"].astype(str).str.lower(),
            "development": frame["msci_development"].astype(str),
            "geo_region": frame["region"].astype(str),
        }
    ).drop_duplicates("location")


# Kenneth French data library ---------------------------------------------------------------

_MONTH_ROW = re.compile(r"^\s*(\d{6})\s*,(.*)$")


def read_french_monthly(path: Path) -> pd.DataFrame:
    """The monthly block of a French CSV zip: the first table after its header row."""
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        text = archive.read(member).decode("latin-1")
    lines = text.splitlines()
    header_at = next(
        (i for i, line in enumerate(lines) if line.startswith(",") and line.count(",") >= 1),
        None,
    )
    if header_at is None:
        raise SchemaError(f"{path.name}: no header row starting with a comma")
    names = [c.strip() for c in lines[header_at].split(",")[1:]]
    rows: list[list[float]] = []
    months: list[str] = []
    for line in lines[header_at + 1 :]:
        match = _MONTH_ROW.match(line)
        if match is None:
            break
        months.append(match.group(1))
        rows.append([float(v) for v in match.group(2).split(",")])
    if not rows:
        raise SchemaError(f"{path.name}: no monthly rows under the header")
    frame = pd.DataFrame(np.asarray(rows), columns=names)
    frame.insert(0, "month", pd.to_datetime(months, format="%Y%m"))
    return frame.replace([-99.99, -999.0], np.nan)
