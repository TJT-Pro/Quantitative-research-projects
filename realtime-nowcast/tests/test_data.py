"""Sources, formats, the real-time panel and the downloader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import requests
from hypothesis import given, settings
from hypothesis import strategies as st
from openpyxl import Workbook

from realtime_nowcast import config as config_mod
from realtime_nowcast.config import SeriesConfig
from realtime_nowcast.data import build, fetch, fixture, fred, gdpnow, vintages
from realtime_nowcast.data.download import Client, mirror_path
from realtime_nowcast.data.manifest import Entry, Manifest
from realtime_nowcast.paths import ProjectPaths

ROOT = Path(__file__).resolve().parents[1]
MONTHLY = SeriesConfig(
    id="RSAFS",
    name="Retail Sales",
    frequency="m",
    transform="pch",
    blocks=("global",),
    category="c",
)
QUARTERLY = SeriesConfig(
    id="Q", name="Q", frequency="q", transform="pca", blocks=("global",), category="c"
)


def _cfg(**download: Any) -> config_mod.Config:
    cfg = config_mod.load(path=ROOT / "configs" / "research.yaml")
    if download:
        cfg = cfg.model_copy(update={"download": cfg.download.model_copy(update=download)})
    return cfg


def _rows(*items: tuple[str, str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(items, columns=["realtime_start", "realtime_end", "date", "value"])


# ALFRED -------------------------------------------------------------------------------------


def test_alfred_page_and_errors() -> None:
    page = {
        "count": 1,
        "observations": [
            {
                "realtime_start": "2011-08-05",
                "realtime_end": "9999-12-31",
                "date": "2011-07-01",
                "value": "131086",
            }
        ],
    }
    rows, count = fred.parse_page(json.dumps(page))
    assert count == 1
    assert rows[0]["value"] == "131086"
    with pytest.raises(fred.SchemaError, match="Bad Request"):
        fred.parse_page(
            json.dumps({"error_code": 400, "error_message": "Bad Request. Variable api_key"})
        )
    with pytest.raises(fred.SchemaError, match="missing"):
        fred.parse_page(json.dumps({"count": 1, "observations": [{"date": "2011-07-01"}]}))
    with pytest.raises(fred.SchemaError, match="not JSON"):
        fred.parse_page("<html>")


def test_urls_carry_no_key() -> None:
    cfg = _cfg()
    for url in (
        fred.vintages_url(cfg, "PAYEMS", offset=0),
        fred.current_url(cfg, "DGS2", offset=0, start="2010-01-01"),
    ):
        assert "api_key" not in url
        assert "file_type=json" in url


def test_pages_are_walked_and_counted(tmp_path: Path) -> None:
    cfg = _cfg(page_limit=3)
    rows = [
        {
            "realtime_start": "2011-01-01",
            "realtime_end": "9999-12-31",
            "date": f"2010-{m:02d}-01",
            "value": str(m),
        }
        for m in range(1, 9)
    ]
    mirror = fixture.Mirror(root=tmp_path / "mirror")
    fixture._put_pages(mirror, rows, url_for=fred.vintages_pages(cfg, "PAYEMS"), limit=3)
    client = Client(
        data_dir=tmp_path / "data", manifest=Manifest(), settings=cfg.download, mirror=mirror.root
    )
    got, first = fetch._pages(
        client, fred.vintages_pages(cfg, "PAYEMS"), secret={}, pause=lambda: None
    )
    assert [r["value"] for r in got] == [str(m) for m in range(1, 9)]
    assert first.endswith("offset=0&file_type=json")


class _Boom:
    headers: ClassVar[dict[str, str]] = {}

    def get(self, url: str, **_: Any) -> Any:
        raise requests.ConnectionError(f"failed to reach {url}")


class _Refuse:
    headers: ClassVar[dict[str, str]] = {}

    def get(self, url: str, **_: Any) -> Any:
        response = requests.Response()
        response.status_code = 400
        response._content = (
            b'{"error_code": 400, "error_message": "Bad Request. The api_key is not registered."}'
        )
        response.url = url
        return response


@pytest.mark.parametrize("session", [_Boom, _Refuse])
def test_the_key_never_reaches_an_error_message(tmp_path: Path, session: type) -> None:
    cfg = _cfg(retries=1)
    client = Client(
        data_dir=tmp_path, manifest=Manifest(), settings=cfg.download, sleep=lambda _: None
    )
    client._local.session = session()
    with pytest.raises(RuntimeError) as caught:
        client.get_text(fred.vintages_url(cfg, "PAYEMS", offset=0), secret={"api_key": "SECRET123"})
    assert "SECRET123" not in str(caught.value)


def test_fetch_without_a_key_explains_how_to_get_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg()
    monkeypatch.delenv(cfg.sources.fred_key_env, raising=False)
    paths = ProjectPaths(root=tmp_path)
    with pytest.raises(RuntimeError, match="fredaccount"):
        fetch.fetch_all(config=cfg, paths=paths, say=lambda _: None)


def test_entries_outside_the_configuration_are_dropped(tmp_path: Path) -> None:
    cfg = _cfg()
    manifest = Manifest()
    for key, url in (
        ("alfred/PAYEMS", "https://api.stlouisfed.org/fred/x"),
        ("alfred/OLD", "https://api.stlouisfed.org/fred/y"),
        ("tardis/x", "https://datasets.tardis.dev/x"),
    ):
        path = tmp_path / "raw" / f"{key.replace('/', '_')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        manifest.put(
            key,
            Entry(
                url=url,
                path=str(path.relative_to(tmp_path)),
                sha256="0",
                bytes=2,
                retrieved_at="t",
                origin="official",
            ),
        )
    assert (
        fetch.drop_retired(
            manifest, data_dir=tmp_path, hosts=cfg.sources.official_hosts, say=lambda _: None
        )
        == 1
    )
    assert fetch.drop_unlisted(manifest, data_dir=tmp_path, config=cfg, say=lambda _: None) == 1
    assert set(manifest.entries) == {"alfred/PAYEMS"}


# The real-time panel ------------------------------------------------------------------------


def _series() -> tuple[vintages.SeriesVintages, vintages.SeriesVintages]:
    monthly = vintages.from_rows(
        MONTHLY,
        _rows(
            ("2011-02-04", "2011-03-03", "2011-01-01", "100.0"),
            ("2011-03-04", "9999-12-31", "2011-01-01", "100.5"),
            ("2011-03-04", "9999-12-31", "2011-02-01", "101.0"),
            ("2011-04-01", "9999-12-31", "2011-03-01", "."),
        ),
    )
    quarterly = vintages.from_rows(
        QUARTERLY,
        _rows(
            ("2011-01-28", "9999-12-31", "2010-10-01", "1000.0"),
            ("2011-04-28", "2011-05-26", "2011-01-01", "1005.0"),
            ("2011-05-27", "9999-12-31", "2011-01-01", "1004.0"),
        ),
    )
    return monthly, quarterly


def test_panel_holds_what_was_published_that_day() -> None:
    series = _series()
    first = vintages.month_index(pd.Period("2010-10", freq="M"))
    last = vintages.month_index(pd.Period("2011-03", freq="M"))

    def on(day: str) -> np.ndarray:
        return vintages.levels_on(series, day=np.datetime64(day), first=first, last=last).levels

    jan = 3  # 2011-01 is row 3 of the grid from 2010-10
    assert np.isnan(on("2011-02-03")[jan, 0])  # not yet released
    assert on("2011-02-04")[jan, 0] == 100.0
    assert on("2011-03-04")[jan, 0] == 100.5  # revised that day
    assert np.isnan(on("2011-04-02")[jan + 2, 0])  # "." stays missing
    # Quarterly values sit on the third month of their quarter.
    assert on("2011-02-01")[2, 1] == 1000.0
    assert np.isnan(on("2011-04-27")[5, 1])
    assert on("2011-04-28")[5, 1] == 1005.0
    assert on("2011-06-01")[5, 1] == 1004.0


@settings(deadline=None, max_examples=60)
@given(offset=st.integers(min_value=0, max_value=4000))
def test_no_value_published_after_the_day_is_used(*, offset: int) -> None:
    """Rebuild the panel on a random day of the fixture's history and look every value up in
    the raw rows: each must have been published on or before that day."""
    rng = np.random.default_rng(offset)
    periods = pd.period_range("2000-01", "2011-12", freq="M")
    x = rng.normal(0.2, 0.5, size=len(periods))
    levels = fixture._levels(MONTHLY, x)
    revised = levels * (1 + rng.normal(0, 0.001, size=levels.size))
    raw = pd.DataFrame(
        fixture._alfred_rows(
            MONTHLY, periods, levels=levels, revised=revised, today=pd.Timestamp("2012-03-01")
        )
    )
    series = (vintages.from_rows(MONTHLY, raw),)
    day = np.datetime64("2000-01-01") + np.timedelta64(offset, "D")
    first = vintages.month_index(periods[0])
    panel = vintages.levels_on(series, day=day, first=first, last=vintages.month_index(periods[-1]))
    starts = pd.to_datetime(raw["realtime_start"]).to_numpy().astype("datetime64[D]")
    ends = (
        pd.to_datetime(raw["realtime_end"].replace("9999-12-31", "2262-04-11"))
        .to_numpy()
        .astype("datetime64[D]")
    )
    for row in np.flatnonzero(~np.isnan(panel.levels[:, 0])):
        date = periods[int(row)].start_time.strftime("%Y-%m-%d")
        match = (raw["date"] == date).to_numpy() & (starts <= day) & (ends >= day)
        assert match.sum() == 1
        assert starts[match][0] <= day
        assert float(raw.loc[match, "value"].iloc[0]) == pytest.approx(panel.levels[row, 0])


def test_advance_estimates_use_the_vintage_of_the_release() -> None:
    _, q = _series()
    adv = build.advance_estimates(q)
    row = adv.loc[adv["quarter"] == "2011Q1"].iloc[0]
    assert row["release"] == pd.Timestamp("2011-04-28")
    assert row["advance"] == pytest.approx(100 * ((1005.0 / 1000.0) ** 4 - 1))


def test_release_days() -> None:
    days = vintages.release_days(_series(), after=np.datetime64("2011-03-01"))
    assert [str(d) for d in days] == ["2011-03-04", "2011-04-01", "2011-04-28", "2011-05-27"]


# GDPNow -------------------------------------------------------------------------------------


def _workbook(tmp_path: Path) -> Path:
    nowcasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-11-01", "2013-12-02", "2014-05-01", "2014-06-02"]),
            "quarter": ["2013Q4", "2013Q4", "2014Q2", "2014Q2"],
            "nowcast": [1.5, 1.8, 2.9, 3.1],
        }
    )
    advance = pd.DataFrame(
        {
            "quarter": ["2013Q4", "2014Q2"],
            "release": pd.to_datetime(["2014-01-30", "2014-07-30"]),
            "advance": [3.2, 4.0],
        }
    )
    path = tmp_path / "gdpnow.xlsx"
    path.write_bytes(fixture._workbook(nowcasts, advance))
    return path


def test_gdpnow_archives_are_read_by_column_name(tmp_path: Path) -> None:
    path = _workbook(tmp_path)
    frame = gdpnow.read_nowcasts(path)
    assert list(frame["quarter"].astype(str)) == ["2013Q4", "2013Q4", "2014Q2", "2014Q2"]
    assert list(frame["nowcast"]) == [1.5, 1.8, 2.9, 3.1]
    record = gdpnow.read_track_record(path)
    assert list(record["advance"]) == [3.2, 4.0]


def test_gdpnow_missing_column_is_named(tmp_path: Path) -> None:
    book = Workbook()
    for name in gdpnow.ARCHIVES:
        book.create_sheet(name).append(["Forecast Date", "Quarter being forecasted"])
    path = tmp_path / "bad.xlsx"
    book.save(path)
    with pytest.raises(gdpnow.SchemaError, match="GDP Nowcast"):
        gdpnow.read_nowcasts(path)


def test_mirror_paths_keep_the_query() -> None:
    url = "https://api.stlouisfed.org/fred/series/observations?series_id=PAYEMS&offset=0"
    path = mirror_path(Path("m"), url)
    assert "__query__" in str(path)
    assert "series_id-PAYEMS" in path.name


def test_gdpnow_falls_back_to_the_next_configured_url(tmp_path: Path) -> None:
    cfg = _cfg()
    mirror = fixture.Mirror(root=tmp_path / "mirror")
    mirror.put(cfg.sources.gdpnow[1], b"workbook bytes")
    manifest = Manifest()
    client = Client(
        data_dir=tmp_path / "data", manifest=manifest, settings=cfg.download, mirror=mirror.root
    )
    summary = fetch.FetchSummary()
    fetch._gdpnow(cfg, client, summary=summary, say=lambda _: None)
    assert manifest.entries[fetch.GDPNOW_KEY].url == cfg.sources.gdpnow[1]
    empty = Client(
        data_dir=tmp_path / "other",
        manifest=Manifest(),
        settings=cfg.download,
        mirror=tmp_path / "none",
    )
    with pytest.raises(RuntimeError, match="GDPNow Model Data"):
        fetch._gdpnow(cfg, empty, summary=summary, say=lambda _: None)
