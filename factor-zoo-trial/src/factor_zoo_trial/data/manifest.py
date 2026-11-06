"""SHA-256 manifest of every raw file: where it came from, when, and what it hashed to."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Origin = Literal["official", "mirror"]


@dataclass(frozen=True)
class Entry:
    url: str
    path: str  # relative to data/raw
    sha256: str
    bytes: int
    retrieved_at: str
    origin: Origin
    etag: str | None = None
    last_modified: str | None = None


def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class Manifest:
    def __init__(self, entries: dict[str, Entry] | None = None) -> None:
        self.entries: dict[str, Entry] = dict(entries or {})

    @classmethod
    def read(cls, path: Path) -> Manifest:
        if not path.is_file():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls({key: Entry(**value) for key, value in payload["files"].items()})

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"files": {k: asdict(v) for k, v in sorted(self.entries.items())}}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def digest(self) -> str:
        """One hash over every file hash, so results.json pins the exact inputs."""
        joined = "\n".join(f"{k}:{v.sha256}" for k, v in sorted(self.entries.items()))
        return hashlib.sha256(joined.encode()).hexdigest()

    def origins(self) -> set[Origin]:
        return {entry.origin for entry in self.entries.values()}
