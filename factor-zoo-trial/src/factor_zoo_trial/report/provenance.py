"""Provenance guard: results built from the synthetic fixture never pass as official."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from factor_zoo_trial.data.manifest import Manifest, sha256_of


def _official(url: str, hosts: tuple[str, ...]) -> bool:
    host = urlsplit(url).netloc.lower()
    return any(host == h or host.endswith("." + h) for h in hosts)


def verify(
    *,
    results: dict[str, Any],
    manifest: Manifest,
    raw_dir: Path,
    official_hosts: tuple[str, ...],
    expected_fingerprint: str | None = None,
    require_raw: bool = True,
) -> list[str]:
    """Every reason results.json is not an official build, or an empty list.

    With `require_raw=False` (CI, where data/raw is never committed) the raw files are not
    re-hashed; everything recorded in results.json and data/manifest.json still is.
    """
    problems: list[str] = []
    if results.get("provenance") != "official":
        problems.append(f"results.json provenance is {results.get('provenance')!r}, not 'official'")
    if not manifest.entries:
        problems.append("data/manifest.json is missing or empty")
    if results.get("manifest_digest") != manifest.digest():
        problems.append("results.json was not built from the files in data/manifest.json")
    if (
        expected_fingerprint is not None
        and results.get("config_fingerprint") != expected_fingerprint
    ):
        problems.append("results.json was built with a different or overridden configuration")
    copied = sorted(k for k, e in manifest.entries.items() if e.origin != "official")
    if copied:
        problems.append(
            f"{len(copied)} of {len(manifest.entries)} raw files have origin 'mirror' "
            f"(for example {copied[0]}), not a download from the source"
        )
    for key, entry in sorted(manifest.entries.items()):
        if not _official(entry.url, official_hosts):
            problems.append(f"{key}: {entry.url} is not on an official host")
        if not require_raw:
            continue
        path = raw_dir / entry.path
        if not path.is_file():
            problems.append(f"{key}: raw file {entry.path} is missing")
        elif sha256_of(path) != entry.sha256:
            problems.append(f"{key}: raw file {entry.path} no longer matches its SHA-256")
    return problems
