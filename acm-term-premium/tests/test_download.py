"""Download recovery: dropped connections, resume, servers ignoring ranges, 304s."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from acm_term_premium.config import DataConfig, ResearchConfig
from acm_term_premium.data.source import SourceError, download, fetch_all, load_manifest

URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"


def _data(config: ResearchConfig, attempts: int = 4) -> DataConfig:
    return config.data.model_copy(
        update={"download_attempts": attempts, "retry_backoff_seconds": 0.0}
    )


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
                200, headers={"content-length": str(len(body))}, stream=_DroppingStream(body[:cut])
            )
        start = int(request.headers["range"].removeprefix("bytes=").rstrip("-"))
        return httpx.Response(
            206,
            headers={"content-range": f"bytes {start}-{len(body) - 1}/{len(body)}"},
            content=body[start:],
        )

    destination = tmp_path / "feds200628.csv"
    download(URL, destination, _data(config), transport=httpx.MockTransport(handler))
    assert destination.read_bytes() == body
    assert ranges == [None, f"bytes={cut}-"], "the second attempt asks only for the missing bytes"
    assert not (tmp_path / "feds200628.csv.part").exists()


def test_restarts_when_server_ignores_range(tmp_path: Path, config: ResearchConfig) -> None:
    body = _payload(50_000)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200, headers={"content-length": str(len(body))}, stream=_DroppingStream(body[:7])
            )
        return httpx.Response(200, content=body)  # range ignored: the whole file again

    destination = tmp_path / "f.csv"
    download(URL, destination, _data(config), transport=httpx.MockTransport(handler))
    assert destination.read_bytes() == body


def test_truncated_body_is_never_installed(tmp_path: Path, config: ResearchConfig) -> None:
    body = _payload(10_000)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-length": str(len(body))}, stream=_DroppingStream(body[:100])
        )

    destination = tmp_path / "f.csv"
    destination.write_bytes(b"previous good copy")
    with pytest.raises(SourceError, match="after 3 attempts"):
        download(URL, destination, _data(config, 3), transport=httpx.MockTransport(handler))
    assert destination.read_bytes() == b"previous good copy"
    assert not (tmp_path / "f.csv.part").exists()


def test_server_errors_are_retried(tmp_path: Path, config: ResearchConfig) -> None:
    body = _payload(1_000)
    statuses = iter([503, 429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, content=body if status == 200 else b"")

    destination = tmp_path / "f.csv"
    download(URL, destination, _data(config), transport=httpx.MockTransport(handler))
    assert destination.read_bytes() == body


def test_refuses_plain_http(tmp_path: Path, config: ResearchConfig) -> None:
    with pytest.raises(SourceError, match="non-https"):
        download("http://example.org/f.csv", tmp_path / "f.csv", config.data)


def test_size_cap_is_enforced(tmp_path: Path, config: ResearchConfig) -> None:
    data = _data(config).model_copy(update={"max_download_bytes": 100})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_payload(1_000))

    with pytest.raises(SourceError, match="above the configured cap"):
        download(URL, tmp_path / "f.csv", data, transport=httpx.MockTransport(handler))


def test_unchanged_files_are_not_downloaded_again(tmp_path: Path, config: ResearchConfig) -> None:
    data = _data(config)
    files = {source.file: f"body of {name}\n".encode() for name, source in data.sources.items()}
    downloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        name = next(
            f
            for f in files
            if request.url.path.endswith(f) or request.url.params.get("id", "") + ".csv" == f
        )
        if request.headers.get("if-none-match") == f'"{name}"':
            return httpx.Response(304)
        downloads.append(name)
        return httpx.Response(200, content=files[name], headers={"etag": f'"{name}"'})

    raw, manifest = tmp_path / "raw", tmp_path / "manifest.json"
    transport = httpx.MockTransport(handler)
    first = fetch_all(data, raw, manifest, transport=transport)
    assert sorted(first.changed) == sorted(name for name, _ in data.sources.items())
    assert sorted(downloads) == sorted(files)

    downloads.clear()
    second = fetch_all(data, raw, manifest, transport=transport)
    assert downloads == [], "every file answered 304 and the local copy was kept"
    assert second.changed == []
    assert set(load_manifest(manifest)) == {name for name, _ in data.sources.items()}

    # A corrupted local copy is not trusted: it is fetched again without conditions.
    victim = next(iter(files))
    (raw / victim).write_bytes(b"corrupt")
    downloads.clear()
    fetch_all(data, raw, manifest, transport=transport)
    assert downloads == [victim]
