"""Resumable downloads with retry, backoff and conditional GET, recorded in the manifest.

A transfer that drops resumes from the bytes already on disk with a Range request,
guarded by If-Range so a file that changed upstream restarts instead of being spliced.
A file already on disk whose hash matches the manifest is revalidated with
If-None-Match / If-Modified-Since, so an unchanged source costs one 304.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import requests

from factor_zoo_trial.config import DownloadConfig
from factor_zoo_trial.data.manifest import Entry, Manifest, sha256_of

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_TRANSIENT = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class RetryableError(RuntimeError):
    """A failure worth another attempt: a dropped stream or a 5xx/429 response."""


class IncompleteError(RetryableError):
    """The stream ended before the advertised length."""


@dataclass(frozen=True)
class Target:
    key: str
    url: str
    relpath: str


@dataclass(frozen=True)
class FetchResult:
    entry: Entry
    changed: bool


def mirror_path(mirror: Path, url: str) -> Path:
    """Where a mirror folder keeps the copy of `url`: <mirror>/<host>/<decoded path>."""
    parts = urlsplit(url)
    return mirror / parts.netloc / unquote(parts.path).lstrip("/")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Fetcher:
    def __init__(
        self,
        *,
        raw_dir: Path,
        manifest: Manifest,
        settings: DownloadConfig,
        mirror: Path | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.raw_dir = raw_dir
        self.manifest = manifest
        self.settings = settings
        self.mirror = mirror
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "factor-zoo-trial/1.0 (research pipeline)")
        self._sleep = sleep

    def fetch(self, target: Target) -> FetchResult:
        dest = self.raw_dir / target.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        previous = self.manifest.entries.get(target.key)
        if self.mirror is not None:
            entry = self._copy_from_mirror(target, dest)
        else:
            entry = self._download(target, dest, previous)
        changed = previous is None or previous.sha256 != entry.sha256
        self.manifest.entries[target.key] = entry
        return FetchResult(entry=entry, changed=changed)

    # Mirror -------------------------------------------------------------------------------

    def _copy_from_mirror(self, target: Target, dest: Path) -> Entry:
        if self.mirror is None:
            raise RuntimeError("no mirror configured")
        source = mirror_path(self.mirror, target.url)
        if not source.is_file():
            raise FileNotFoundError(f"mirror has no copy of {target.url} (expected {source})")
        shutil.copyfile(source, dest)
        return Entry(
            url=target.url,
            path=target.relpath,
            sha256=sha256_of(dest),
            bytes=dest.stat().st_size,
            retrieved_at=_now(),
            origin="mirror",
        )

    # HTTP ---------------------------------------------------------------------------------

    def _download(self, target: Target, dest: Path, previous: Entry | None) -> Entry:
        conditional: dict[str, str] = {}
        if previous is not None and dest.is_file() and sha256_of(dest) == previous.sha256:
            if previous.etag:
                conditional["If-None-Match"] = previous.etag
            if previous.last_modified:
                conditional["If-Modified-Since"] = previous.last_modified
        last_error: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                outcome = self._attempt(target, dest, conditional)
            except (RetryableError, *_TRANSIENT) as error:
                last_error = error
                if attempt == self.settings.retries:
                    break
                self._sleep(self._delay(attempt, error))
                continue
            if outcome is None:  # 304: the copy on disk is current
                if previous is None:
                    raise RuntimeError(f"{target.url}: 304 without a previous copy")
                return replace(previous, retrieved_at=_now())
            return outcome
        raise RuntimeError(
            f"{target.url}: gave up after {self.settings.retries + 1} attempts: {last_error}"
        )

    def _delay(self, attempt: int, error: Exception) -> float:
        retry_after = getattr(error, "retry_after", None)
        if isinstance(retry_after, float):
            return retry_after
        return float(self.settings.backoff_seconds * (2**attempt))

    def _attempt(self, target: Target, dest: Path, conditional: dict[str, str]) -> Entry | None:
        part = dest.with_name(dest.name + ".part")
        meta = dest.with_name(dest.name + ".part.json")
        offset = part.stat().st_size if part.is_file() else 0
        headers = dict(conditional)
        part_validator = _read_validator(meta) if offset else None
        if offset and part_validator:
            headers = {"Range": f"bytes={offset}-", "If-Range": part_validator}
        elif offset:
            offset = 0  # no validator for the partial bytes: they cannot be trusted
        with self.session.get(
            target.url, headers=headers, stream=True, timeout=self.settings.timeout_seconds
        ) as response:
            status = response.status_code
            if status == 304:
                return None
            if status == 416:
                part.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
                raise RetryableError("range not satisfiable; restarting from zero")
            if status in RETRYABLE_STATUS:
                error = RetryableError(f"HTTP {status}")
                header = response.headers.get("Retry-After", "")
                if header.isdigit():
                    error.retry_after = float(header)  # type: ignore[attr-defined]
                raise error
            if status not in (200, 206):
                raise RuntimeError(f"{target.url}: HTTP {status}")
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
            if status == 200:
                offset = 0
                _write_validator(meta, etag or last_modified)
            expected = _expected_total(response, offset)
            mode = "ab" if status == 206 else "wb"
            with part.open(mode) as handle:
                for block in response.iter_content(chunk_size=self.settings.chunk_bytes):
                    handle.write(block)
        size = part.stat().st_size
        if expected is not None and size != expected:
            raise IncompleteError(f"received {size} of {expected} bytes")
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
