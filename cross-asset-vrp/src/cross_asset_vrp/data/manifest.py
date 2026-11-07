"""SHA-256 manifest of every raw file: where it came from, when, and what it hashed to.

The manifest is written atomically, so an interrupted run never leaves it half-written.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

Origin = Literal["official", "mirror"]


@dataclass(frozen=True)
class Entry:
    url: str
    path: str  # relative to data/
    sha256: str
    bytes: int
    retrieved_at: str
    origin: Origin
    etag: str | None = None
    last_modified: str | None = None


def official(url: str, hosts: tuple[str, ...]) -> bool:
    """True when `url` is on one of `hosts` or a subdomain of one."""
    host = urlsplit(url).netloc.lower()
    return any(host == h or host.endswith("." + h) for h in hosts)


def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_of(path: Path, *, chunk: int = 1 << 20) -> str:
    """S3 ETags of single-part uploads are the MD5 of the content."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class Manifest:
    def __init__(self, entries: dict[str, Entry] | None = None) -> None:
        self.entries: dict[str, Entry] = dict(entries or {})
        self._lock = threading.Lock()

    @classmethod
    def read(cls, path: Path) -> Manifest:
        if not path.is_file():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls({key: Entry(**value) for key, value in payload["files"].items()})

    def put(self, key: str, entry: Entry) -> None:
        with self._lock:
            self.entries[key] = entry

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {"files": {k: asdict(v) for k, v in sorted(self.entries.items())}}
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        temporary.replace(path)

    def digest(self) -> str:
        joined = "\n".join(f"{k}:{v.sha256}" for k, v in sorted(self.entries.items()))
        return hashlib.sha256(joined.encode()).hexdigest()

    def origins(self) -> set[Origin]:
        return {entry.origin for entry in self.entries.values()}
