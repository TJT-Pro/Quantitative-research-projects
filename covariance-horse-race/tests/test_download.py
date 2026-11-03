"""Download recovery: dropped connections, resume, servers ignoring ranges, 304s."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from covariance_horse_race.config import ResearchConfig
from covariance_horse_race.data.source import SourceError, download, fetch_all, load_manifest

URL = "https://example.org/ftp/file.zip"


def _config(config: ResearchConfig, attempts: int = 4) -> ResearchConfig:
    data = config.data.model_copy(
        update={"download_attempts": attempts, "retry_backoff_seconds": 0.0}
    )
    return config.model_copy(update={"data": data})


class _DroppingStream(httpx.SyncByteStream):
    """Sends part of the body, then the peer closes the connection."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self) -> Iterator[bytes]:
        yield self.data
        raise httpx.RemoteProtocolError("peer closed connection without sending complete body")


def _payload(size: int) -> bytes:
    return bytes(i % 251 for i in range(size))


def test_resumes_after_dropped_connection(tmp_path: Path, config: ResearchConfig) -> None:
    body = _payload(200_000)
    cut = 60_000
    ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ranges.append(request.headers.get("range"))
        if len(ranges) == 1:
            return httpx.Response(
                200,
                headers={"content-length": str(len(body))},
                stream=_DroppingStream(body[:cut]),
            )
        start = int(request.headers["range"].removeprefix("bytes=").rstrip("-"))
        return httpx.Response(
            206,
            headers={"content-range": f"bytes {start}-{len(body) - 1}/{len(body)}"},
            content=body[start:],
        )

    destination = tmp_path / "file.zip"
    download(URL, destination, _config(config).data, transport=httpx.MockTransport(handler))
    assert destination.read_bytes() == body
    assert ranges == [None, f"bytes={cut}-"], "second attempt asks only for the missing bytes"
    assert not (tmp_path / "file.zip.part").exists()


def test_restarts_when_server_ignores_range(tmp_path: Path, config: ResearchConfig) -> None:
    body = _payload(50_000)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200, headers={"content-length": str(len(body))}, stream=_DroppingStream(body[:7])
            )
        return httpx.Response(200, content=body)  # range ignored: whole file again

    destination = tmp_path / "file.zip"
    download(URL, destination, _config(config).data, transport=httpx.MockTransport(handler))
    assert destination.read_bytes() == body


def test_truncated_body_is_never_installed(tmp_path: Path, config: ResearchConfig) -> None:
    body = _payload(10_000)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-length": str(len(body))}, stream=_DroppingStream(body[:100])
        )

    destination = tmp_path / "file.zip"
    destination.write_bytes(b"previous good copy")
    with pytest.raises(SourceError, match="after 3 attempts"):
        download(URL, destination, _config(config, 3).data, transport=httpx.MockTransport(handler))
    assert destination.read_bytes() == b"previous good copy"
    assert not (tmp_path / "file.zip.part").exists()


def test_server_errors_are_retried(tmp_path: Path, config: ResearchConfig) -> None:
    body = _payload(1_000)
    statuses = iter([503, 502, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, content=body if status == 200 else b"")

    destination = tmp_path / "file.zip"
    download(URL, destination, _config(config).data, transport=httpx.MockTransport(handler))
    assert destination.read_bytes() == body


def test_refuses_plain_http(tmp_path: Path, config: ResearchConfig) -> None:
    with pytest.raises(SourceError, match="non-https"):
        download("http://example.org/f.zip", tmp_path / "f.zip", config.data)


def _zip_bytes(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.csv", text)
    return buffer.getvalue()


def test_unchanged_files_are_not_downloaded_again(tmp_path: Path, config: ResearchConfig) -> None:
    cfg = _config(config)
    files = {name: _zip_bytes(f"{name}\n,A\n20200101,1.0\n") for name in cfg.data.files.values()}
    page = "".join(f'<a href="ftp/{name}">x</a>' for name in files)
    downloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".html"):
            return httpx.Response(200, text=page)
        name = request.url.path.rsplit("/", 1)[1]
        if request.headers.get("if-none-match") == f'"{name}"':
            return httpx.Response(304)
        downloads.append(name)
        return httpx.Response(200, content=files[name], headers={"etag": f'"{name}"'})

    raw, manifest = tmp_path / "raw", tmp_path / "manifest.json"
    raw.mkdir()
    transport = httpx.MockTransport(handler)
    first = fetch_all(cfg.data, raw, manifest, transport=transport)
    assert sorted(first.changed) == sorted(cfg.data.files)
    assert len(downloads) == len(files)

    downloads.clear()
    second = fetch_all(cfg.data, raw, manifest, transport=transport)
    assert downloads == [], "every file answered 304 and was kept"
    assert second.changed == []
    assert set(load_manifest(manifest)) == set(cfg.data.files)

    # A corrupted local copy is not trusted: it is fetched again unconditionally.
    victim = next(iter(files))
    (raw / victim).write_bytes(b"corrupt")
    downloads.clear()
    fetch_all(cfg.data, raw, manifest, transport=transport)
    assert downloads == [victim]
