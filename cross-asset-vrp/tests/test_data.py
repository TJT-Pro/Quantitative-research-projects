"""Downloader failure modes, the bucket listing and every parser on its exact source format."""

from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from cross_asset_vrp import config as config_mod
from cross_asset_vrp.config import DownloadConfig
from cross_asset_vrp.data import cboe, crypto, fetch, fixture, listing, prices
from cross_asset_vrp.data.download import Client, MissingError, Target, mirror_path
from cross_asset_vrp.data.manifest import Entry, Manifest

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = DownloadConfig(
    retries=4,
    backoff_seconds=0.01,
    timeout_seconds=5.0,
    chunk_bytes=1024,
    workers=2,
    rest_pause_seconds=0.0,
)


@dataclass
class Behaviour:
    payload: bytes = bytes(range(256)) * 300
    etag: str = '"v1"'
    fail_first: int = 0
    drop_after: int | None = None
    status: int | None = None  # answer every request with this status
    log: list[dict[str, str]] = field(default_factory=list)


class _Handler(BaseHTTPRequestHandler):
    behaviour: Behaviour

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        b = self.behaviour
        b.log.append(dict(self.headers.items()))
        if b.status is not None:
            self.send_response(b.status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if b.fail_first > 0:
            b.fail_first -= 1
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.headers.get("If-None-Match") == b.etag:
            self.send_response(304)
            self.end_headers()
            return
        start = 0
        if self.headers.get("Range") and self.headers.get("If-Range") == b.etag:
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
        body = b.payload[start:]
        self.send_response(206 if start else 200)
        self.send_header("ETag", b.etag)
        self.send_header("Content-Length", str(len(body)))
        if start:
            self.send_header(
                "Content-Range", f"bytes {start}-{len(b.payload) - 1}/{len(b.payload)}"
            )
        self.end_headers()
        if b.drop_after is not None and not start:
            self.wfile.write(body[: b.drop_after])
            self.wfile.flush()
            b.drop_after = None
            self.connection.close()
            return
        self.wfile.write(body)


@pytest.fixture
def server() -> Iterator[tuple[str, Behaviour]]:
    behaviour = Behaviour()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (_Handler,), {"behaviour": behaviour}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/f.zip", behaviour
    httpd.shutdown()


def _client(tmp_path: Path, manifest: Manifest | None = None) -> Client:
    return Client(
        data_dir=tmp_path, manifest=manifest or Manifest(), settings=SETTINGS, sleep=lambda _: None
    )


def test_resumes_after_a_dropped_connection(server: tuple[str, Behaviour], tmp_path: Path) -> None:
    url, b = server
    b.drop_after = 20_000
    _client(tmp_path).fetch(Target(key="k", url=url, relpath="f.zip"))
    assert (tmp_path / "f.zip").read_bytes() == b.payload
    assert any("Range" in h for h in b.log)


def test_retries_server_errors(server: tuple[str, Behaviour], tmp_path: Path) -> None:
    url, b = server
    b.fail_first = 3
    _client(tmp_path).fetch(Target(key="k", url=url, relpath="f.zip"))
    assert (tmp_path / "f.zip").read_bytes() == b.payload


def test_a_matching_listing_md5_skips_the_request(
    server: tuple[str, Behaviour], tmp_path: Path
) -> None:
    url, b = server
    md5 = hashlib.md5(b.payload, usedforsecurity=False).hexdigest()
    manifest = Manifest()
    target = Target(key="k", url=url, relpath="f.zip", md5=md5)
    _client(tmp_path, manifest).fetch(target)
    b.log.clear()
    again = _client(tmp_path, manifest).fetch(target)
    assert not again.requested
    assert b.log == []


def test_a_wrong_md5_is_refused(server: tuple[str, Behaviour], tmp_path: Path) -> None:
    url, _ = server
    with pytest.raises(RuntimeError, match="MD5"):
        _client(tmp_path).fetch(Target(key="k", url=url, relpath="f.zip", md5="0" * 32))
    assert not (tmp_path / "f.zip").exists()


def test_files_without_a_listing_are_revalidated(
    server: tuple[str, Behaviour], tmp_path: Path
) -> None:
    url, b = server
    manifest = Manifest()
    target = Target(key="k", url=url, relpath="f.zip")
    _client(tmp_path, manifest).fetch(target)
    b.log.clear()
    second = _client(tmp_path, manifest).fetch(target)
    assert not second.changed
    assert b.log[0].get("If-None-Match") == '"v1"'


def test_mirror_paths_keep_the_query() -> None:
    path = mirror_path(Path("m"), "https://h.example/a/b?x=1&y=2")
    assert path == Path("m/h.example/a/b/__query__/x-1__y-2")


XML = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>b</Name><Prefix>p/</Prefix>
<Marker></Marker>{next}<IsTruncated>{truncated}</IsTruncated>
<Contents><Key>p/A-1h-2024-01.zip</Key><Size>10</Size><ETag>"0123456789abcdef0123456789abcdef"</ETag></Contents>
<Contents><Key>p/A-1h-2024-02.zip</Key><Size>10</Size><ETag>"abc-2"</ETag></Contents>
<CommonPrefixes><Prefix>p/Z/</Prefix></CommonPrefixes></ListBucketResult>"""


def test_listing_page_with_and_without_next_marker() -> None:
    page, truncated, marker = listing.parse_page(
        XML.format(next="<NextMarker>p/Q</NextMarker>", truncated="true")
    )
    assert truncated
    assert marker == "p/Q"
    assert page.prefixes == ["p/Z/"]
    assert page.objects[0].md5 == "0123456789abcdef0123456789abcdef"
    assert page.objects[1].md5 is None  # a multipart ETag is not an MD5
    _, _, fallback = listing.parse_page(XML.format(next="", truncated="true"))
    assert fallback == "p/Z/"  # the largest key or prefix on the page


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_cboe_index_in_both_layouts(tmp_path: Path) -> None:
    ohlc = _write(
        tmp_path,
        "VIX.csv",
        "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,17.24,17.24,17.24,17.24\n01/03/1990,18.19,18.5,18.0,18.19\n",
    )
    single = _write(
        tmp_path, "GVZ.csv", "DATE,GVZ\n09/18/2009,22.620000\n09/21/2009,23.220000\n09/22/2009,.\n"
    )
    a = cboe.read_index(ohlc, symbol="VIX")
    b = cboe.read_index(single, symbol="GVZ")
    assert a["value"].tolist() == [17.24, 18.19]
    assert a["date"].iloc[1] == pd.Timestamp("1990-01-03")
    assert b["value"].tolist() == [22.62, 23.22]  # the unparseable row is dropped


def test_cboe_index_without_a_value_column_fails(tmp_path: Path) -> None:
    bad = _write(tmp_path, "X.csv", "DATE,OTHER\n01/02/1990,1\n")
    with pytest.raises(cboe.SchemaError):
        cboe.read_index(bad, symbol="X")


FUTURES_HEADER = (
    "Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest"
)


def test_vx_files_new_and_archive_layouts(tmp_path: Path) -> None:
    new = _write(
        tmp_path,
        "new.csv",
        FUTURES_HEADER + "\n2022-04-25,F (Jan 2023),0.0000,26.0000,28.5000,0.0000,27.25,0,0,0,0\n"
        "2023-01-18,F (Jan 2023),19.5500,19.7000,18.9500,19.0900,19.09,-0.2255,958,0,22982\n",
    )
    old = _write(
        tmp_path,
        "old.csv",
        "CFE data\n" + FUTURES_HEADER + "\n08/19/2005,K (May 06),0.00,0.00,0.00,0.00,0.00,,0,0,0\n"
        "08/22/2005,K (May 06),163.50,165.80,163.50,165.80,165.40,165.40,30,0,30,\n"
        "08/23/2005,K (May 06),165.00,166.00,164.00,165.90,165.90,0.50,12,0,31,,\n",
    )
    a = cboe.read_futures(new, expiry=date(2023, 1, 18))
    b = cboe.read_futures(old, expiry=date(2006, 5, 17))
    assert a["settle"].tolist() == [27.25, 19.09]
    # A zero settle is no settlement; trailing commas past the header are dropped.
    assert b["settle"].tolist() == [165.4, 165.9]
    assert b["trade_date"].iloc[0] == pd.Timestamp("2005-08-22")


def test_vx_row_with_a_value_past_the_header_is_refused(tmp_path: Path) -> None:
    bad = _write(
        tmp_path,
        "bad.csv",
        FUTURES_HEADER + "\n08/22/2005,K (May 06),163.50,165.80,163.50,165.80,165.40,0,30,0,30,7\n",
    )
    with pytest.raises(cboe.SchemaError, match="line 2"):
        cboe.read_futures(bad, expiry=date(2006, 5, 17))


@pytest.mark.parametrize(
    ("year", "month", "expected"),
    [
        (2023, 1, date(2023, 1, 18)),
        (2008, 1, date(2008, 1, 16)),
        (2006, 5, date(2006, 5, 17)),
        (2014, 3, date(2014, 3, 18)),  # Good Friday 2014 falls 30 days later
        (2019, 4, date(2019, 4, 17)),
    ],
)
def test_vx_expiry_rule_matches_published_dates(year: int, month: int, expected: date) -> None:
    assert cboe.monthly_expiry(year, month) == expected


def test_vx_urls_switch_to_the_archive_before_2013() -> None:
    cfg = config_mod.load(path=ROOT / "configs" / "research.yaml")
    assert cboe.futures_url(cfg, date(2023, 1, 18)).endswith("/VX/VX_2023-01-18.csv")
    assert cboe.futures_url(cfg, date(2008, 1, 16)).endswith("/CFE_F08_VX.csv")


def test_tiingo_rows_and_errors(tmp_path: Path) -> None:
    row = {
        "date": "2020-01-02T00:00:00.000Z",
        "close": 10.0,
        "high": 11.0,
        "low": 9.0,
        "open": 10.0,
        "volume": 1,
        "adjClose": 5.0,
        "adjHigh": 5.5,
        "adjLow": 4.5,
        "adjOpen": 5.0,
        "adjVolume": 2,
        "divCash": 0.0,
        "splitFactor": 1.0,
    }
    good = _write(tmp_path, "SPY.json", json.dumps([row]))
    frame = prices.read_tiingo(good, ticker="SPY")
    assert frame[["open", "high", "low", "close"]].iloc[0].tolist() == [
        5.0,
        5.5,
        4.5,
        5.0,
    ]  # adjusted
    error = _write(tmp_path, "BAD.json", json.dumps({"detail": "Invalid token."}))
    with pytest.raises(prices.SchemaError, match="Invalid token"):
        prices.read_tiingo(error, ticker="BAD")


def test_fred_old_and_new_headers(tmp_path: Path) -> None:
    for column in ("observation_date", "DATE"):
        path = _write(tmp_path, f"{column}.csv", f"{column},DTB3\n2020-01-02,1.52\n2020-01-03,.\n")
        frame = prices.read_fred(path, series="DTB3")
        assert frame["rate"].tolist() == [pytest.approx(0.0152)]


def test_tiingo_key_is_required_only_for_live_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config_mod.load(path=ROOT / "configs" / "research.yaml")
    monkeypatch.delenv(cfg.sources.tiingo_key_env, raising=False)
    assert prices.tiingo_headers(cfg, required=False) == {}
    with pytest.raises(RuntimeError, match=cfg.sources.tiingo_key_env):
        prices.tiingo_headers(cfg, required=True)
    monkeypatch.setenv(cfg.sources.tiingo_key_env, "abc")
    assert prices.tiingo_headers(cfg, required=True)["Authorization"] == "Token abc"
    assert "abc" not in prices.tiingo_url(cfg, "SPY")


def _zip(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "m.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("m.csv", text)
    return path


def test_minute_klines_with_and_without_header_and_in_microseconds(tmp_path: Path) -> None:
    rows = (
        "1704067200000,1,2,0.5,1.5,1,1704067259999,1,1,1,1,0\n"
        "1704067260000,1.5,2,1,1.8,1,1704067319999,1,1,1,1,0\n"
    )
    plain = crypto.read_minutes(_zip(tmp_path, rows))
    headed = crypto.read_minutes(_zip(tmp_path, ",".join(crypto.KLINE_COLUMNS) + "\n" + rows))
    micro = crypto.read_minutes(
        _zip(
            tmp_path,
            rows.replace("1704067200000,", "1704067200000000,").replace(
                "1704067260000,", "1704067260000000,"
            ),
        )
    )
    for frame in (plain, headed, micro):
        assert frame["time"].iloc[1] == pd.Timestamp("2024-01-01 00:01", tz="UTC")
        assert frame["close"].tolist() == [1.5, 1.8]


def test_dvol_page_and_errors() -> None:
    page = json.dumps(
        {"jsonrpc": "2.0", "result": {"data": [[1, 50, 51, 49, 50.5]], "continuation": 0}}
    )
    data, continuation = crypto.parse_dvol_page(page)
    assert data == [[1, 50, 51, 49, 50.5]]
    assert continuation == 0
    with pytest.raises(crypto.SchemaError):
        crypto.parse_dvol_page(json.dumps({"error": {"message": "bad"}}))


def _trade(n: int, stamp: int) -> dict[str, object]:
    return {
        "trade_id": f"t{n}",
        "instrument_name": "BTC-29MAR24-60000-C",
        "timestamp": stamp,
        "price": 0.02,
        "mark_price": 0.019,
        "index_price": 60000.0,
        "direction": "buy",
        "amount": 1.0,
        "iv": 60.0,
    }


def test_trade_page_and_errors() -> None:
    page = json.dumps({"result": {"trades": [_trade(1, 5)], "has_more": True}})
    trades, more = crypto.parse_trades_page(page)
    assert more
    assert trades[0]["trade_id"] == "t1"
    with pytest.raises(crypto.SchemaError, match="without a result"):
        crypto.parse_trades_page(json.dumps({"error": {"code": 10}}))
    broken = {k: v for k, v in _trade(1, 5).items() if k != "mark_price"}
    with pytest.raises(crypto.SchemaError, match="mark_price"):
        crypto.parse_trades_page(json.dumps({"result": {"trades": [broken], "has_more": False}}))


def test_trade_window_pages_forward_and_drops_repeats(tmp_path: Path) -> None:
    cfg = config_mod.load(path=ROOT / "configs" / "research.yaml")
    cfg = cfg.model_copy(
        update={"replication": cfg.replication.model_copy(update={"page_size": 10})}
    )
    day = pd.Timestamp("2024-03-01", tz="UTC")
    start = int((day - pd.Timedelta(hours=cfg.replication.window_hours)).value // 1_000_000)
    # 35 trades, three at a time on the same millisecond, so pages overlap at their edges.
    trades = [_trade(n, start + 1000 * (n // 3)) for n in range(35)]
    mirror = fixture.Mirror(root=tmp_path / "mirror")
    fixture.trade_pages(cfg, mirror, day=day, trades=list(trades))
    client = Client(
        data_dir=tmp_path / "data", manifest=Manifest(), settings=cfg.download, mirror=mirror.root
    )
    result = fetch.trades_window(cfg, client, day, pause=lambda: None)
    saved = json.loads((tmp_path / "data" / result.entry.path).read_text(encoding="utf-8"))
    assert [t["trade_id"] for t in saved] == [t["trade_id"] for t in trades]
    assert set(saved[0]) == set(crypto.TRADE_FIELDS)  # only the fields the rebuild reads
    again = fetch.trades_window(cfg, client, day, pause=lambda: None)
    assert not again.requested


def test_a_full_page_on_one_millisecond_is_refused(tmp_path: Path) -> None:
    cfg = config_mod.load(path=ROOT / "configs" / "research.yaml")
    cfg = cfg.model_copy(
        update={"replication": cfg.replication.model_copy(update={"page_size": 10})}
    )
    day = pd.Timestamp("2024-03-01", tz="UTC")
    start = int((day - pd.Timedelta(hours=cfg.replication.window_hours)).value // 1_000_000)
    mirror = fixture.Mirror(root=tmp_path / "mirror")
    fixture.trade_pages(cfg, mirror, day=day, trades=[_trade(n, start) for n in range(25)])
    client = Client(
        data_dir=tmp_path / "data", manifest=Manifest(), settings=cfg.download, mirror=mirror.root
    )
    with pytest.raises(RuntimeError, match="one millisecond"):
        fetch.trades_window(cfg, client, day, pause=lambda: None)


def test_a_403_counts_as_missing_only_where_the_source_says_so(
    server: tuple[str, Behaviour], tmp_path: Path
) -> None:
    url, b = server
    b.status = 403
    with pytest.raises(RuntimeError, match="HTTP 403"):
        _client(tmp_path).fetch(Target(key="k", url=url, relpath="f.zip"))
    with pytest.raises(MissingError):
        _client(tmp_path).fetch(
            Target(key="k", url=url, relpath="f.zip", missing_statuses=(403, 404))
        )


def test_vx_gaps_before_the_monthly_cycle_and_after_the_last_listing_are_expected() -> None:
    months = [date(2004 + (m // 12), m % 12 + 1, 15) for m in range(36)]
    present = set(months[3:30]) - {months[5]}  # early gap, then monthly, then not listed yet
    assert cboe.gaps_inside_listing(months, present) == []
    present.discard(months[20])  # a failed download inside the cycle
    assert cboe.gaps_inside_listing(months, present) == [months[20]]


def test_fetch_drops_files_from_retired_sources(tmp_path: Path) -> None:
    kept = tmp_path / "raw" / "cboe" / "VIX.csv"
    gone = tmp_path / "raw" / "old" / "chain.csv.gz"
    for path in (kept, gone):
        path.parent.mkdir(parents=True)
        path.write_text("x", encoding="utf-8")
    manifest = Manifest()
    for key, url, path in (
        ("cboe/VIX", "https://cdn.cboe.com/VIX.csv", kept),
        ("old/chain", "https://datasets.example.dev/chain.csv.gz", gone),
    ):
        manifest.put(
            key,
            Entry(
                url=url,
                path=str(path.relative_to(tmp_path)),
                sha256="0",
                bytes=1,
                retrieved_at="2026-01-01T00:00:00Z",
                origin="official",
            ),
        )
    removed = fetch.drop_retired(
        manifest, data_dir=tmp_path, hosts=("cdn.cboe.com",), say=lambda _: None
    )
    assert removed == 1
    assert set(manifest.entries) == {"cboe/VIX"}
    assert kept.exists()
    assert not gone.exists()


def test_vx_files_held_in_both_places_are_merged() -> None:
    days = pd.to_datetime(["2012-11-01", "2012-11-02", "2013-01-02", "2013-01-03"])
    archive = pd.DataFrame({"trade_date": days, "expiry": days[-1], "settle": [1.0, 2, 3, 4]})
    primary = pd.DataFrame({"trade_date": days[2:], "expiry": days[-1], "settle": [30.0, 40.0]})
    merged = cboe.merge_sources(primary, archive)
    assert merged["trade_date"].tolist() == list(days)
    assert merged["settle"].tolist() == [1.0, 2.0, 30.0, 40.0]


def test_longest_gap_finds_a_missing_stretch() -> None:
    dates = pd.Series(pd.to_datetime(["2012-11-20", "2012-11-21", "2013-05-21", "2013-05-22"]))
    assert cboe.longest_gap(dates) == (181, pd.Timestamp("2013-05-21"))
    assert cboe.longest_gap(dates.iloc[:1]) == (0, None)


def test_gaps_lists_every_long_stretch() -> None:
    dates = pd.Series(
        pd.to_datetime(["2004-07-13", "2004-07-14", "2004-07-22", "2004-10-13", "2004-10-22"])
    )
    found = cboe.gaps(dates, longer_than=7)
    assert [(b.date().isoformat(), a.date().isoformat(), d) for b, a, d in found] == [
        ("2004-07-14", "2004-07-22", 8),
        ("2004-07-22", "2004-10-13", 83),
        ("2004-10-13", "2004-10-22", 9),
    ]
    assert cboe.gaps(dates.iloc[:2], longer_than=7) == []
