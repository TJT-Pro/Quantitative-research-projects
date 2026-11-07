"""HTTP access with retry, backoff, resume and conditional GET, or a mirror folder.

- A file whose listing ETag (the MD5 of a single-part S3 object) matches the manifest and
  the copy on disk is skipped without a request.
- Otherwise a partial download resumes with Range, guarded by If-Range so a file that
  changed upstream restarts instead of being spliced, and the result is checked against
  the listing MD5 when there is one.
- A file without a listing ETag is revalidated with If-None-Match / If-Modified-Since.

With a mirror, every URL (query string included) maps to a file under the mirror folder,
so the synthetic fixture drives exactly the same code path offline.

Secrets (the FRED key) travel as `secret` query parameters added only when the request is
sent: URLs in the manifest, the mirror and every error message are the ones without them,
and exceptions from the HTTP library, whose text can carry the full URL, are reported by
type only.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlencode, urlsplit

import requests

from realtime_nowcast.config import DownloadConfig
from realtime_nowcast.data.manifest import Entry, Manifest, Origin, md5_of, sha256_of

RETRYABLE_STATUS = frozenset({408, 418, 425, 429, 500, 502, 503, 504})
_TRANSIENT = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class RetryableError(RuntimeError):
    """Worth another attempt: a dropped stream, a 5xx or a rate limit."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class MissingError(FileNotFoundError):
    """The source answered 404: the file does not exist upstream."""


@dataclass(frozen=True)
class Target:
    key: str
    url: str
    relpath: str  # relative to data/
    md5: str | None = None  # from the bucket listing, when known
    headers: tuple[tuple[str, str], ...] = ()  # sent with the request, never recorded
    # Statuses that mean "this file does not exist". An S3 bucket that does not allow listing
    # answers 403 instead of 404 for a missing key, so such sources add 403 here.
    missing_statuses: tuple[int, ...] = (404,)


@dataclass(frozen=True)
class FetchResult:
    entry: Entry
    changed: bool
    requested: bool


def mirror_path(mirror: Path, url: str) -> Path:
    """<mirror>/<host>/<decoded path>[/__query__/<query with = and & made filename-safe>]."""
    parts = urlsplit(url)
    path = mirror / parts.netloc / unquote(parts.path).lstrip("/")
    if parts.query:
        safe = unquote(parts.query).replace("&", "__").replace("=", "-").replace("/", "_")
        path = path / "__query__" / safe
    return path


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Client:
    """One client per run. Sessions are per thread: requests.Session is not thread-safe."""

    def __init__(
        self,
        *,
        data_dir: Path,
        manifest: Manifest,
        settings: DownloadConfig,
        mirror: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.data_dir = data_dir
        self.manifest = manifest
        self.settings = settings
        self.mirror = mirror
        self._sleep = sleep
        self._local = threading.local()

    @property
    def origin(self) -> Origin:
        return "mirror" if self.mirror is not None else "official"

    def _session(self) -> requests.Session:
        session: requests.Session | None = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = "realtime-nowcast/1.0 (research pipeline)"
            self._local.session = session
        return session

    # Text requests (listings, REST) ---------------------------------------------------------

    def get_text(self, url: str, *, secret: dict[str, str] | None = None) -> str:
        if self.mirror is not None:
            source = mirror_path(self.mirror, url)
            if not source.is_file():
                raise MissingError(f"mirror has no copy of {url} (expected {source})")
            return source.read_text(encoding="utf-8")
        full = url + ("&" if "?" in url else "?") + urlencode(secret) if secret else url
        last = ""
        for attempt in range(self.settings.retries + 1):
            try:
                response = self._session().get(full, timeout=self.settings.timeout_seconds)
                status = response.status_code
                if status == 404:
                    raise MissingError(url)
                if status in RETRYABLE_STATUS:
                    header = response.headers.get("Retry-After", "")
                    raise RetryableError(
                        f"HTTP {status}", retry_after=float(header) if header.isdigit() else None
                    )
                if status >= 400:
                    raise RuntimeError(f"{url}: HTTP {status}: {_api_message(response)}")
                return response.text
            except RetryableError as error:
                last = str(error)
                delay = self._delay(attempt, error)
            except _TRANSIENT as error:
                last = type(error).__name__  # its text can hold the URL with the secret
                delay = self._delay(attempt, error)
            if attempt < self.settings.retries:
                self._sleep(delay)
        raise RuntimeError(f"{url}: gave up after {self.settings.retries + 1} attempts: {last}")

    def save_text(self, *, key: str, url: str, relpath: str, text: str) -> FetchResult:
        """Record a REST response as a raw file with its own manifest entry."""
        dest = self.data_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        previous = self.manifest.entries.get(key)
        entry = Entry(
            url=url,
            path=relpath,
            sha256=sha256_of(dest),
            bytes=dest.stat().st_size,
            retrieved_at=_now(),
            origin="mirror" if self.mirror is not None else "official",
        )
        self.manifest.put(key, entry)
        changed = previous is None or previous.sha256 != entry.sha256
        return FetchResult(entry=entry, changed=changed, requested=True)

    # Files ----------------------------------------------------------------------------------

    def fetch(self, target: Target) -> FetchResult:
        dest = self.data_dir / target.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        previous = self.manifest.entries.get(target.key)
        if (
            previous is not None
            and target.md5 is not None
            and previous.etag == target.md5
            and dest.is_file()
            and sha256_of(dest) == previous.sha256
        ):
            return FetchResult(entry=previous, changed=False, requested=False)
        if self.mirror is not None:
            entry = self._copy(target, dest)
        else:
            entry = self._download(target, dest, previous)
        if target.md5 is not None and md5_of(dest) != target.md5:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"{target.url}: MD5 does not match the bucket listing")
        if target.md5 is not None:
            entry = replace(entry, etag=target.md5)
        self.manifest.put(target.key, entry)
        changed = previous is None or previous.sha256 != entry.sha256
        return FetchResult(entry=entry, changed=changed, requested=True)

    def _copy(self, target: Target, dest: Path) -> Entry:
        if self.mirror is None:
            raise RuntimeError("no mirror configured")
        source = mirror_path(self.mirror, target.url)
        if not source.is_file():
            raise MissingError(f"mirror has no copy of {target.url} (expected {source})")
        shutil.copyfile(source, dest)
        return Entry(
            url=target.url,
            path=target.relpath,
            sha256=sha256_of(dest),
            bytes=dest.stat().st_size,
            retrieved_at=_now(),
            origin="mirror",
        )

    def _delay(self, attempt: int, error: Exception) -> float:
        if isinstance(error, RetryableError) and error.retry_after is not None:
            return error.retry_after
        return float(self.settings.backoff_seconds * (2**attempt))

    def _download(self, target: Target, dest: Path, previous: Entry | None) -> Entry:
        conditional: dict[str, str] = {}
        if (
            target.md5 is None
            and previous is not None
            and dest.is_file()
            and sha256_of(dest) == previous.sha256
        ):
            if previous.etag:
                conditional["If-None-Match"] = previous.etag
            if previous.last_modified:
                conditional["If-Modified-Since"] = previous.last_modified
        last: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                outcome = self._attempt(target, dest, conditional)
            except (RetryableError, *_TRANSIENT) as error:
                last = error
                if attempt < self.settings.retries:
                    self._sleep(self._delay(attempt, error))
                continue
            if outcome is None:
                if previous is None:
                    raise RuntimeError(f"{target.url}: 304 without a previous copy")
                return replace(previous, retrieved_at=_now())
            return outcome
        raise RuntimeError(
            f"{target.url}: gave up after {self.settings.retries + 1} attempts: {last}"
        )

    def _attempt(self, target: Target, dest: Path, conditional: dict[str, str]) -> Entry | None:
        part = dest.with_name(dest.name + ".part")
        meta = dest.with_name(dest.name + ".part.json")
        offset = part.stat().st_size if part.is_file() else 0
        validator = _read_validator(meta) if offset else None
        headers = dict(target.headers) | dict(conditional)
        if offset and validator:
            headers = dict(target.headers) | {"Range": f"bytes={offset}-", "If-Range": validator}
        elif offset:
            offset = 0
        with self._session().get(
            target.url, headers=headers, stream=True, timeout=self.settings.timeout_seconds
        ) as response:
            status = response.status_code
            if status == 304:
                return None
            if status in target.missing_statuses:
                raise MissingError(f"{target.url}: HTTP {status}, no such file")
            if status == 416:
                part.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
                raise RetryableError("range not satisfiable; restarting")
            if status in RETRYABLE_STATUS:
                header = response.headers.get("Retry-After", "")
                raise RetryableError(
                    f"HTTP {status}", retry_after=float(header) if header.isdigit() else None
                )
            if status not in (200, 206):
                raise RuntimeError(f"{target.url}: HTTP {status}")
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
            if status == 200:
                offset = 0
                _write_validator(meta, etag or last_modified)
            expected = _expected_total(response, offset)
            with part.open("ab" if status == 206 else "wb") as handle:
                for block in response.iter_content(chunk_size=self.settings.chunk_bytes):
                    handle.write(block)
        size = part.stat().st_size
        if expected is not None and size != expected:
            raise RetryableError(f"received {size} of {expected} bytes")
        part.replace(dest)
        meta.unlink(missing_ok=True)
        return Entry(
            url=target.url,
            path=target.relpath,
            sha256=sha256_of(dest),
            bytes=size,
            retrieved_at=_now(),
            origin="official",
            etag=etag,
            last_modified=last_modified,
        )


def _api_message(response: requests.Response) -> str:
    """The API's own error text (FRED answers JSON with error_message), never the URL."""
    try:
        payload = response.json()
    except ValueError:
        return response.reason or "error"
    if isinstance(payload, dict) and "error_message" in payload:
        return str(payload["error_message"])
    return response.reason or "error"


def _expected_total(response: requests.Response, offset: int) -> int | None:
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        if total.isdigit():
            return int(total)
    length = response.headers.get("Content-Length", "")
    if length.isdigit() and "Content-Encoding" not in response.headers:
        return offset + int(length)
    return None


def _write_validator(meta: Path, validator: str | None) -> None:
    if validator:
        meta.write_text(json.dumps({"validator": validator}), encoding="utf-8")
    else:
        meta.unlink(missing_ok=True)


def _read_validator(meta: Path) -> str | None:
    if not meta.is_file():
        return None
    value = json.loads(meta.read_text(encoding="utf-8")).get("validator")
    return value if isinstance(value, str) else None
