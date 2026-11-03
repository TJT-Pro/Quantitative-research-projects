"""Download and verification of the library's zip files.

Raw files are immutable once written. Every download is recorded in a manifest
with its URL, retrieval time, size and SHA-256, so a silent revision of the
source (the library switched CRSP vintages in January 2025) is detected rather
than absorbed.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from covariance_horse_race.config import DataConfig

# A legitimate CSV in these archives compresses roughly 3-6x; anything far beyond
# that is treated as a decompression bomb.
_MAX_COMPRESSION_RATIO = 100.0
_MAX_UNCOMPRESSED_BYTES = 1_000_000_000
_USER_AGENT = "covariance-horse-race/1.0 (academic research pipeline; python-httpx)"


class SourceError(RuntimeError):
    """The upstream source is unavailable, changed shape, or failed validation."""


class _LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.links.append(value)


def zip_links(page_html: str, page_url: str) -> dict[str, str]:
    """Map each linked zip file name to its absolute URL."""
    collector = _LinkCollector()
    collector.feed(page_html)
    found: dict[str, str] = {}
    for href in collector.links:
        absolute = urljoin(page_url, href)
        name = Path(urlparse(absolute).path).name
        if name.lower().endswith(".zip"):
            found[name] = absolute
    return found


def check_files_listed(config: DataConfig, links: dict[str, str]) -> None:
    """Fail if a configured file is no longer linked, naming the nearest links."""
    problems: list[str] = []
    for key, name in config.files.items():
        if name not in links:
            close = difflib.get_close_matches(name, links, n=3, cutoff=0.6)
            hint = f" closest links: {close}" if close else ""
            problems.append(f"{key}: {name!r} is not linked on the library page.{hint}")
    if problems:
        raise SourceError("\n".join(problems))


@dataclass(frozen=True)
class ManifestEntry:
    key: str
    file: str
    url: str
    retrieved_at: str
    bytes: int
    sha256: str
    csv_member: str
    csv_sha256: str
    # HTTP validators from the server, used to skip unchanged files on the next fetch.
    etag: str | None = None
    last_modified: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _client(config: DataConfig, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        timeout=config.request_timeout_seconds,
        follow_redirects=True,
        # Identity encoding keeps byte ranges aligned with the file on disk,
        # which resuming an interrupted download depends on.
        headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"},
        transport=transport,
    )


def fetch_library_links(
    config: DataConfig, transport: httpx.BaseTransport | None = None
) -> dict[str, str]:
    url = str(config.library_page)
    last_error: Exception | None = None
    with _client(config, transport) as client:
        for attempt in range(1, config.download_attempts + 1):
            try:
                response = client.get(url)
                if response.status_code < 500:
                    response.raise_for_status()
                    return zip_links(response.text, url)
                last_error = SourceError(f"server answered {response.status_code}")
            except httpx.TransportError as exc:
                last_error = exc
            if attempt < config.download_attempts:
                time.sleep(config.retry_backoff_seconds * 2 ** (attempt - 1))
    raise SourceError(f"library page unreachable: {last_error}") from last_error


class _RetryableError(Exception):
    """A failure worth retrying: dropped connection, truncated body, server error."""


@dataclass(frozen=True)
class DownloadResult:
    not_modified: bool
    etag: str | None = None
    last_modified: str | None = None


def _expected_total(response: httpx.Response, already: int) -> int | None:
    content_range = response.headers.get("content-range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        return int(total) if total.isdigit() else None
    length = response.headers.get("content-length")
    return already + int(length) if length and length.isdigit() else None


def _attempt(
    client: httpx.Client,
    url: str,
    part: Path,
    config: DataConfig,
    conditional: dict[str, str],
) -> DownloadResult:
    have = part.stat().st_size if part.is_file() else 0
    headers = {"Range": f"bytes={have}-"} if have else dict(conditional)
    with client.stream("GET", url, headers=headers) as response:
        final = str(response.url)
        if urlparse(final).scheme != "https":
            raise SourceError(f"{url} redirected to non-https {final}")
        if response.status_code == 304:
            return DownloadResult(not_modified=True)
        if response.status_code == 416:
            # The partial file no longer matches the server's file: start over.
            part.unlink(missing_ok=True)
            raise _RetryableError("range not satisfiable, restarting")
        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableError(f"server answered {response.status_code}")
        response.raise_for_status()
        if have and response.status_code != 206:
            have = 0  # the server ignored the range and is sending the whole file
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
        return DownloadResult(
            not_modified=False,
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
        )


def download(
    url: str,
    destination: Path,
    config: DataConfig,
    *,
    conditional: dict[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    log: Callable[[str], None] | None = None,
) -> DownloadResult:
    """Download `url` to `destination`, resuming and retrying on transient failures.

    Bytes go to `<destination>.part`; after a dropped connection the next attempt
    asks for the remaining range only. The file replaces `destination` only once
    the full declared length has arrived. With `conditional` validators
    (If-None-Match / If-Modified-Since) an unchanged file is not downloaded.
    """
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


def read_single_csv(zip_path: Path) -> tuple[str, bytes]:
    """Return (member name, bytes) of the one CSV inside a library archive.

    Rejects archives with more or fewer than one CSV, absolute or parent-relative
    member paths, and members whose size or compression ratio indicates a bomb.
    Nothing is extracted to disk.
    """
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise SourceError(f"{zip_path.name} is not a valid zip archive") from exc
    with archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        csvs = [m for m in members if m.filename.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise SourceError(f"{zip_path.name}: expected one CSV member, found {len(csvs)}")
        member = csvs[0]
        parts = Path(member.filename).parts
        if member.filename.startswith(("/", "\\")) or ".." in parts:
            raise SourceError(f"{zip_path.name}: unsafe member path {member.filename!r}")
        if member.file_size > _MAX_UNCOMPRESSED_BYTES:
            raise SourceError(f"{zip_path.name}: member larger than allowed")
        ratio = member.file_size / max(member.compress_size, 1)
        if ratio > _MAX_COMPRESSION_RATIO:
            raise SourceError(f"{zip_path.name}: compression ratio {ratio:.0f} looks like a bomb")
        payload = archive.read(member)
        if len(payload) != member.file_size:
            raise SourceError(f"{zip_path.name}: member size mismatch")
    return member.filename, payload


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: ManifestEntry(**entry) for key, entry in raw["files"].items()}


def write_manifest(path: Path, entries: dict[str, ManifestEntry]) -> None:
    payload = {
        "source": "Kenneth R. French Data Library",
        "files": {key: asdict(entry) for key, entry in sorted(entries.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class FetchReport:
    entries: dict[str, ManifestEntry]
    changed: list[str]


def _conditional(previous: ManifestEntry | None, destination: Path) -> dict[str, str]:
    """Validators for a conditional GET, only when the local copy is intact."""
    if previous is None or not destination.is_file() or _sha256(destination) != previous.sha256:
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
    mirror: Path | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    log: Callable[[str], None] | None = None,
) -> FetchReport:
    """Download every configured file and update the manifest after each one.

    Files the server reports as unchanged are kept. The manifest is written after
    every file, so an interrupted run keeps what it already has. With `mirror`,
    files are copied from a local folder laid out like the library's ftp folder
    (CI uses this with a generated fixture). Returns the keys whose content changed.
    """
    previous = load_manifest(manifest_path)
    if mirror is None:
        links = fetch_library_links(config, transport)
        check_files_listed(config, links)
    entries = {k: v for k, v in previous.items() if k in config.files}
    changed: list[str] = []
    for key, name in sorted(config.files.items()):
        destination = raw_dir / name
        etag = last_modified = None
        if mirror is not None:
            source = mirror / name
            if not source.is_file():
                raise SourceError(f"mirror has no {name}")
            url = source.resolve().as_uri()
            fd, tmp_name = tempfile.mkstemp(dir=raw_dir, suffix=".part")
            os.close(fd)
            tmp = Path(tmp_name)
            tmp.write_bytes(source.read_bytes())
            tmp.replace(destination)
        else:
            url = links[name]
            old = previous.get(key)
            if log is not None:
                log(f"{key}: {name}")
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
        member, payload = read_single_csv(destination)
        entry = ManifestEntry(
            key=key,
            file=name,
            url=url,
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            bytes=destination.stat().st_size,
            sha256=_sha256(destination),
            csv_member=member,
            csv_sha256=hashlib.sha256(payload).hexdigest(),
            etag=etag,
            last_modified=last_modified,
        )
        if log is not None and mirror is None:
            log(f"  downloaded {entry.bytes / 1e6:.1f} MB")
        old = previous.get(key)
        if old is None or old.csv_sha256 != entry.csv_sha256:
            changed.append(key)
        entries[key] = entry
        write_manifest(manifest_path, entries)
    write_manifest(manifest_path, entries)
    return FetchReport(entries=entries, changed=changed)


def verify_raw(raw_dir: Path, manifest_path: Path) -> None:
    """Fail if any raw file differs from the hash recorded when it was fetched."""
    entries = load_manifest(manifest_path)
    if not entries:
        raise SourceError("no manifest: run `chr fetch` first")
    for entry in entries.values():
        path = raw_dir / entry.file
        if not path.is_file():
            raise SourceError(f"missing raw file {entry.file}; run `chr fetch`")
        if _sha256(path) != entry.sha256:
            raise SourceError(f"{entry.file} does not match its manifest hash")
