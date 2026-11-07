"""Downloader failure modes, the bucket listing, contract mapping and every parser."""

from __future__ import annotations

import hashlib
import io
import threading
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from perp_funding_carry import config as config_mod
from perp_funding_carry.config import DownloadConfig
from perp_funding_carry.data import listing, parsers, vision
from perp_funding_carry.data.build import Tables, screen
from perp_funding_carry.data.download import Client, Target, mirror_path
from perp_funding_carry.data.manifest import Manifest

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
    log: list[dict[str, str]] = field(default_factory=list)


class _Handler(BaseHTTPRequestHandler):
    behaviour: Behaviour

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        b = self.behaviour
        b.log.append(dict(self.headers.items()))
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


def test_contracts_map_to_spot_with_multipliers() -> None:
    cfg = config_mod.load(path=ROOT / "configs" / "research.yaml")
    perps = {"BTCUSDT", "1000PEPEUSDT", "1000SATSUSDT", "BTCUSDC", "NOSPOTUSDT"}
    spots = {"BTCUSDT", "PEPEUSDT", "1000SATSUSDT"}
    mapped = {c.symbol: (c.spot, c.multiplier) for c in vision.map_contracts(cfg, perps, spots)}
    assert mapped == {
        "BTCUSDT": ("BTCUSDT", 1),
        "1000PEPEUSDT": ("PEPEUSDT", 1000),
        "1000SATSUSDT": ("1000SATSUSDT", 1),
    }


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "x.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("x.csv", text)
    path.write_bytes(buffer.getvalue())
    return path


ROW_MS = "1704067200000,1,2,0.5,1.5,10,1704070799999,15,3,5,7,0"
ROW_US = "1735689600000000,1,2,0.5,1.5,10,1735693199999999,15,3,5,7,0"


def test_klines_without_and_with_a_header(tmp_path: Path) -> None:
    plain = parsers.read_klines(_write(tmp_path, ROW_MS + "\n"))
    assert plain["open_time"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    header = ",".join(parsers.KLINE_COLUMNS) + "\n" + ROW_MS + "\n"
    assert parsers.read_klines(_write(tmp_path, header)).equals(plain)


def test_spot_klines_in_microseconds(tmp_path: Path) -> None:
    frame = parsers.read_klines(_write(tmp_path, ROW_US + "\n"))
    assert frame["open_time"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")


def test_funding_file_header(tmp_path: Path) -> None:
    text = "calc_time,funding_interval_hours,last_funding_rate\n1704096000001,8,8.4E-7\n"
    frame = parsers.read_funding(_write(tmp_path, text))
    assert frame["time"].iloc[0] == pd.Timestamp("2024-01-01 08:00", tz="UTC")
    assert frame["rate"].iloc[0] == pytest.approx(8.4e-7)


def test_a_wrong_layout_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(parsers.SchemaError, match="columns"):
        parsers.read_klines(_write(tmp_path, "1,2,3\n"))
    with pytest.raises(parsers.SchemaError, match="sum_open_interest_value"):
        parsers.read_metrics(_write(tmp_path, "create_time,symbol\n2024-01-01 00:00:00,X\n"))


def test_screen_drops_impossible_prices_and_open_interest() -> None:
    time = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    bars = pd.DataFrame(
        {
            "symbol": "X",
            "open_time": time,
            "mark_close": [1.0, 1.0, 1.0],
            "spot_close": [1.0, 0.0, float("inf")],
        }
    )
    oi = pd.DataFrame({"symbol": "X", "time": time, "oi_usd": [5.0, 0.0, -1.0]})
    stable = pd.DataFrame({"open_time": time, "usd_per_usdt": [1.0, 1.0, 0.0]})
    funding = pd.DataFrame({"symbol": "X", "time": time, "rate": [1e-4, float("nan"), 2e-4]})
    empty = pd.DataFrame()
    tables = Tables(
        contracts=empty,
        funding=funding,
        bars=bars,
        core_bars=bars.iloc[0:0],
        premium=empty,
        oi=oi,
        stable=stable,
        venues=empty,
        funding_info=empty,
    )
    clean, dropped = screen(tables)
    assert dropped == {"bars": 2, "core_bars": 0, "stable": 1, "oi": 2, "funding": 1}
    assert clean.bars["spot_close"].tolist() == [1.0]
    assert clean.oi["oi_usd"].tolist() == [5.0]
