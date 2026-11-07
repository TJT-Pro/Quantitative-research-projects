"""Provenance guard: results built from the synthetic fixture never pass as official.

The default check needs only what the repository commits (results.json and
data/manifest.json), so CI runs it without the raw downloads: results.json must say
'official', its manifest digest must match the committed manifest, every manifest entry must
come from an official host by an official download, and the configuration fingerprint must
match the committed config. With `rehash`, every raw file on disk is hashed again (in
parallel, with progress) and compared with its manifest entry.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cross_asset_vrp.data.manifest import Manifest, official, sha256_of


def verify(
    *,
    results: dict[str, Any],
    manifest: Manifest,
    official_hosts: tuple[str, ...],
    expected_fingerprint: str,
) -> list[str]:
    problems: list[str] = []
    if results.get("provenance") != "official":
        problems.append(f"results.json provenance is {results.get('provenance')!r}, not 'official'")
    if not manifest.entries:
        problems.append("data/manifest.json is missing or empty")
    if results.get("manifest_digest") != manifest.digest():
        problems.append("results.json was not built from the files in data/manifest.json")
    if results.get("config_fingerprint") != expected_fingerprint:
        problems.append("results.json was built with a different or overridden configuration")
    copied = sorted(k for k, e in manifest.entries.items() if e.origin != "official")
    if copied:
        problems.append(
            f"{len(copied)} of {len(manifest.entries)} raw files have origin 'mirror' "
            f"(for example {copied[0]}), not a download from the source"
        )
    off_host = sorted(k for k, e in manifest.entries.items() if not official(e.url, official_hosts))
    if off_host:
        problems.append(
            f"{len(off_host)} of {len(manifest.entries)} raw files are not from an official "
            f"host (for example {off_host[0]})"
        )
    return problems


def rehash(
    *, manifest: Manifest, data_dir: Path, workers: int, say: Callable[[str], None]
) -> list[str]:
    """Hash every raw file again; returns one problem per missing or changed file."""
    entries = sorted(manifest.entries.items())
    problems: list[str] = []
    step = max(1, len(entries) // 20)

    def check(item: tuple[str, Any]) -> str | None:
        key, entry = item
        path = data_dir / entry.path
        if not path.is_file():
            return f"{key}: raw file {entry.path} is missing"
        if sha256_of(path) != entry.sha256:
            return f"{key}: raw file {entry.path} no longer matches its SHA-256"
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check, item) for item in entries]
        for done, future in enumerate(as_completed(futures), start=1):
            problem = future.result()
            if problem:
                problems.append(problem)
            if done % step == 0 or done == len(entries):
                say(f"  hashed {done:,} / {len(entries):,} raw files")
    return problems
