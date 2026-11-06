"""The downloader against a local HTTP server that drops, fails and revalidates."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from factor_zoo_trial.config import DownloadConfig
from factor_zoo_trial.data.download import Fetcher, Target
from factor_zoo_trial.data.manifest import Manifest

SETTINGS = DownloadConfig(retries=4, backoff_seconds=0.01, timeout_seconds=5.0, chunk_bytes=1024)


@dataclass
class Behaviour:
    payload: bytes = b""
    etag: str = '"v1"'
    fail_first: int = 0  # requests answered with 503
    drop_after: int | None = None  # bytes sent before the first full response is cut
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
        range_header = self.headers.get("Range")
        if range_header and self.headers.get("If-Range") == b.etag:
            start = int(range_header.split("=")[1].split("-")[0])
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
    behaviour = Behaviour(payload=bytes(range(256)) * 400)
    handler = type("Handler", (_Handler,), {"behaviour": behaviour})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/file.bin", behaviour
    httpd.shutdown()


def _fetcher(tmp_path: Path, manifest: Manifest) -> Fetcher:
    return Fetcher(raw_dir=tmp_path, manifest=manifest, settings=SETTINGS, sleep=lambda _: None)


def test_resumes_after_a_dropped_connection(server: tuple[str, Behaviour], tmp_path: Path) -> None:
    url, behaviour = server
    behaviour.drop_after = 30_000
    result = _fetcher(tmp_path, Manifest()).fetch(Target(key="k", url=url, relpath="f.bin"))
    assert (tmp_path / "f.bin").read_bytes() == behaviour.payload
    assert result.entry.sha256 == hashlib.sha256(behaviour.payload).hexdigest()
    resumed = [int(h["Range"][6:-1]) for h in behaviour.log if "Range" in h]
    assert resumed
    assert 0 < resumed[0] <= 30_000  # whole chunks written before the drop are kept


def test_retries_server_errors(server: tuple[str, Behaviour], tmp_path: Path) -> None:
    url, behaviour = server
    behaviour.fail_first = 3
    _fetcher(tmp_path, Manifest()).fetch(Target(key="k", url=url, relpath="f.bin"))
    assert (tmp_path / "f.bin").read_bytes() == behaviour.payload


def test_gives_up_after_the_configured_retries(
    server: tuple[str, Behaviour], tmp_path: Path
) -> None:
    url, behaviour = server
    behaviour.fail_first = 10
    with pytest.raises(RuntimeError, match="gave up"):
        _fetcher(tmp_path, Manifest()).fetch(Target(key="k", url=url, relpath="f.bin"))


def test_unchanged_file_is_revalidated_with_one_conditional_request(
    server: tuple[str, Behaviour], tmp_path: Path
) -> None:
    url, behaviour = server
    manifest = Manifest()
    target = Target(key="k", url=url, relpath="f.bin")
    _fetcher(tmp_path, manifest).fetch(target)
    behaviour.log.clear()
    second = _fetcher(tmp_path, manifest).fetch(target)
    assert not second.changed
    assert len(behaviour.log) == 1
    assert behaviour.log[0].get("If-None-Match") == '"v1"'


def test_changed_upstream_file_restarts_a_partial_download(
    server: tuple[str, Behaviour], tmp_path: Path
) -> None:
    url, behaviour = server
    behaviour.drop_after = 10_000
    behaviour.fail_first = 0
    manifest = Manifest()
    fetcher = Fetcher(
        raw_dir=tmp_path,
        manifest=manifest,
        settings=SETTINGS.model_copy(update={"retries": 0}),
        sleep=lambda _: None,
    )
    with pytest.raises(RuntimeError):
        fetcher.fetch(Target(key="k", url=url, relpath="f.bin"))
    assert 0 < (tmp_path / "f.bin.part").stat().st_size <= 10_000
    behaviour.payload = b"new version" * 1000
    behaviour.etag = '"v2"'
    _fetcher(tmp_path, manifest).fetch(Target(key="k", url=url, relpath="f.bin"))
    assert (tmp_path / "f.bin").read_bytes() == behaviour.payload


def test_manifest_round_trips_and_digest_tracks_content(tmp_path: Path) -> None:
    manifest = Manifest()
    (tmp_path / "mirror" / "example.org").mkdir(parents=True)
    (tmp_path / "mirror" / "example.org" / "a.txt").write_text("a")
    fetcher = Fetcher(
        raw_dir=tmp_path / "raw", manifest=manifest, settings=SETTINGS, mirror=tmp_path / "mirror"
    )
    fetcher.fetch(Target(key="a", url="https://example.org/a.txt", relpath="a.txt"))
    manifest.write(tmp_path / "m.json")
    again = Manifest.read(tmp_path / "m.json")
    assert again.digest() == manifest.digest()
    assert again.origins() == {"mirror"}
