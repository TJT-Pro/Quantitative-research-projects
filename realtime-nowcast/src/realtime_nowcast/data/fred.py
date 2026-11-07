"""FRED and ALFRED: URLs and the JSON they answer.

ALFRED keeps every vintage of a series. series/observations with realtime_start=1776-07-04,
realtime_end=9999-12-31 and output_type=1 answers one row per value and the interval of
days during which that value was the published one:

    {"count": 1234, "offset": 0, "limit": 100000,
     "observations": [{"realtime_start": "2011-08-05", "realtime_end": "2011-09-01",
                       "date": "2011-07-01", "value": "131086"}, ...]}

`date` is the first day of the observation period; "." means no value; "9999-12-31" ends
the interval of a value that is still current. Pages hold `limit` rows and are walked with
`offset`. The key goes in the api_key parameter, which the downloader adds at send time.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from realtime_nowcast.config import Config

REALTIME_ALL = ("1776-07-04", "9999-12-31")
FIELDS = ("realtime_start", "realtime_end", "date", "value")


class SchemaError(ValueError):
    """A response does not have the layout this pipeline was written against."""


def vintages_url(config: Config, series: str, *, offset: int) -> str:
    start, end = REALTIME_ALL
    return (
        f"{config.sources.fred_api}/series/observations?series_id={series}"
        f"&realtime_start={start}&realtime_end={end}&output_type=1"
        f"&observation_start={config.panel.observation_start}&sort_order=asc"
        f"&limit={config.download.page_limit}&offset={offset}&file_type=json"
    )


def current_url(config: Config, series: str, *, offset: int, start: str) -> str:
    """Today's vintage of a series that is not revised (the daily Treasury yields)."""
    return (
        f"{config.sources.fred_api}/series/observations?series_id={series}"
        f"&observation_start={start}&sort_order=asc"
        f"&limit={config.download.page_limit}&offset={offset}&file_type=json"
    )


def vintages_pages(config: Config, series: str) -> Callable[[int], str]:
    """The URL of each page of a series' vintages, by row offset."""

    def url(offset: int) -> str:
        return vintages_url(config, series, offset=offset)

    return url


def current_pages(config: Config, series: str, *, start: str) -> Callable[[int], str]:
    def url(offset: int) -> str:
        return current_url(config, series, offset=offset, start=start)

    return url


def parse_page(text: str) -> tuple[list[dict[str, str]], int]:
    """The rows on one page and the total row count."""
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as error:
        raise SchemaError(f"FRED answered something that is not JSON: {error}") from error
    if not isinstance(payload, dict) or "observations" not in payload:
        message = payload.get("error_message") if isinstance(payload, dict) else None
        raise SchemaError(f"FRED answered without observations: {message or payload}")
    rows = payload["observations"]
    if not isinstance(rows, list):
        raise SchemaError("FRED observations is not a list")
    for row in rows[:1]:
        missing = [f for f in FIELDS if f not in row]
        if missing:
            raise SchemaError(f"FRED rows are missing {missing}; found {sorted(row)}")
    count = payload.get("count")
    if not isinstance(count, int):
        raise SchemaError("FRED answered without a row count")
    return [{f: str(r[f]) for f in FIELDS} for r in rows], count
