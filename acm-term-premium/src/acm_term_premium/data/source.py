"""Downloads with resume, retry and conditional requests, recorded in a manifest.

Each file goes to `<name>.part` first. A dropped connection is retried with
exponential backoff and the retry asks only for the missing byte range. The
file replaces the previous copy only once the full declared length arrived.
Each completed file is hashed into data/manifest.json, and a later run sends the
stored validators so an unchanged file is not downloaded again.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from acm_term_premium.config import DataConfig

_USER_AGENT = "acm-term-premium/1.0 (academic research pipeline; python-httpx)"

Log = Callable[[str], None]


class SourceError(RuntimeError):
    """A source is unreachable, changed shape, or failed validation."""


class _RetryableError(Exception):
    """Dropped connection, truncated body or server error: worth another attempt."""


@dataclass(frozen=True)
class ManifestEntry:
    name: str
    file: str
    url: str
    retrieved_at: str
    bytes: int
    sha256: str
    etag: str | None = None
    last_modified: str | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _client(config: DataConfig, transport: httpx.BaseTransport | None) -> httpx.Client:
    return httpx.Client(
        timeout=config.request_timeout_seconds,
        follow_redirects=True,
        # Identity encoding keeps byte ranges aligned with the file on disk.
        headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"},
        transport=transport,
    )


def _expected_total(response: httpx.Response, already: int) -> int | None:
    content_range = response.headers.get("content-range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        return int(total) if total.isdigit() else None
    length = response.headers.get("content-length")
    return already + int(length) if length and length.isdigit() else None


@dataclass(frozen=True)
class _Result:
    not_modified: bool
    etag: str | None = None
    last_modified: str | None = None


def _attempt(
    client: httpx.Client, url: str, part: Path, config: DataConfig, conditional: dict[str, str]
) -> _Result:
    have = part.stat().st_size if part.is_file() else 0
    headers = {"Range": f"bytes={have}-"} if have else dict(conditional)
    with client.stream("GET", url, headers=headers) as response:
        if urlparse(str(response.url)).scheme != "https":
            raise SourceError(f"{url} redirected to non-https {response.url}")
        if response.status_code == 304:
            return _Result(not_modified=True)
        if response.status_code == 416:
            part.unlink(missing_ok=True)
            raise _RetryableError("range not satisfiable, restarting")
        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableError(f"server answered {response.status_code}")
        response.raise_for_status()
        if have and response.status_code != 206:
            have = 0
        expected = _expected_total(response, have)
        if expected is not None and expected > config.max_download_bytes:
            raise SourceError(f"{url} is {expected} bytes, above the configured cap")
        received = have
        with part.open("ab" if have else "wb") as out:
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > config.max_download_bytes:
                    raise SourceError(f"{url} exceeded {config.max_download_bytes} bytes")
                out.write(chunk)
        if received == 0:
            raise _RetryableError("empty body")
        if expected is not None and received != expected:
            raise _RetryableError(f"received {received} of {expected} bytes")
        return _Result(False, response.headers.get("etag"), response.headers.get("last-modified"))


def download(
    url: str,
    destination: Path,
    config: DataConfig,
    *,
    conditional: dict[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    log: Log | None = None,
) -> _Result:
    if urlparse(url).scheme != "https":
        raise SourceError(f"refusing non-https URL {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    with _client(config, transport) as client:
        for attempt in range(1, config.download_attempts + 1):
            try:
                result = _attempt(client, url, part, config, conditional or {})
            except (httpx.TransportError, _RetryableError) as exc:
                last_error = exc
                if attempt == config.download_attempts:
                    break
                wait = config.retry_backoff_seconds * 2 ** (attempt - 1)
                if log is not None:
                    done = part.stat().st_size if part.is_file() else 0
                    log(
                        f"  {destination.name}: {exc}; retry {attempt} of "
                        f"{config.download_attempts - 1} in {wait:.0f}s, resuming at {done:,} bytes"
                    )
                time.sleep(wait)
                continue
            if not result.not_modified:
                part.replace(destination)
            return result
    part.unlink(missing_ok=True)
    raise SourceError(
        f"{url}: download failed after {config.download_attempts} attempts: {last_error}"
    ) from last_error


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: ManifestEntry(**entry) for name, entry in raw["files"].items()}


def write_manifest(path: Path, entries: dict[str, ManifestEntry]) -> None:
    payload = {"files": {name: asdict(entry) for name, entry in sorted(entries.items())}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class FetchReport:
    entries: dict[str, ManifestEntry]
    changed: list[str]


def _conditional(previous: ManifestEntry | None, destination: Path) -> dict[str, str]:
    if previous is None or not destination.is_file() or sha256(destination) != previous.sha256:
        return {}
    headers = {}
    if previous.etag:
        headers["If-None-Match"] = previous.etag
    if previous.last_modified:
        headers["If-Modified-Since"] = previous.last_modified
    return headers


def fetch_all(
    config: DataConfig,
    raw_dir: Path,
    manifest_path: Path,
    *,
    mirror: Path | None = None,
    transport: httpx.BaseTransport | None = None,
    log: Log | None = None,
) -> FetchReport:
    """Fetch every source; with `mirror`, copy same-named files from a local folder instead."""
    previous = load_manifest(manifest_path)
    entries = {k: v for k, v in previous.items() if k in dict(config.sources.items())}
    changed: list[str] = []
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, source in config.sources.items():
        destination = raw_dir / source.file
        etag = last_modified = None
        if mirror is not None:
            local = mirror / source.file
            if not local.is_file():
                raise SourceError(f"mirror has no {source.file}")
            part = destination.with_name(destination.name + ".part")
            part.write_bytes(local.read_bytes())
            part.replace(destination)
            url = local.resolve().as_uri()
        else:
            url = str(source.url)
            old = previous.get(name)
            if log is not None:
                log(f"{name}: {source.file}")
            result = download(
                url,
                destination,
                config,
                conditional=_conditional(old, destination),
                transport=transport,
                log=log,
            )
            if result.not_modified and old is not None:
                if log is not None:
                    log(f"  unchanged since {old.retrieved_at[:10]}, kept local copy")
                continue
            etag, last_modified = result.etag, result.last_modified
        entry = ManifestEntry(
            name=name,
            file=source.file,
            url=url,
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            bytes=destination.stat().st_size,
            sha256=sha256(destination),
            etag=etag,
            last_modified=last_modified,
        )
        if log is not None and mirror is None:
            log(f"  downloaded {entry.bytes / 1e6:.1f} MB")
        old = previous.get(name)
        if old is None or old.sha256 != entry.sha256:
            changed.append(name)
        entries[name] = entry
        write_manifest(manifest_path, entries)
    write_manifest(manifest_path, entries)
    return FetchReport(entries=entries, changed=changed)


def verify_raw(raw_dir: Path, manifest_path: Path) -> dict[str, ManifestEntry]:
    entries = load_manifest(manifest_path)
    if not entries:
        raise SourceError("no manifest: run `acm fetch` first")
    for entry in entries.values():
        path = raw_dir / entry.file
        if not path.is_file():
            raise SourceError(f"missing raw file {entry.file}; run `acm fetch`")
        if sha256(path) != entry.sha256:
            raise SourceError(f"{entry.file} does not match its manifest hash")
    return entries
