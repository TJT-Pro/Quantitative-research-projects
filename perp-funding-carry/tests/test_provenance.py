"""The provenance guard passes on committed files alone; rehashing catches changed raw files."""

from __future__ import annotations

from pathlib import Path

from perp_funding_carry.data.manifest import Entry, Manifest, sha256_of
from perp_funding_carry.report import provenance

HOSTS = ("binance.vision",)
FINGERPRINT = "abc123"


def _manifest(data: Path) -> Manifest:
    manifest = Manifest()
    for name, body in (("a.zip", b"first"), ("b.zip", b"second")):
        path = data / "raw" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        manifest.put(
            name,
            Entry(
                url=f"https://data.binance.vision/{name}",
                path=f"raw/{name}",
                sha256=sha256_of(path),
                bytes=len(body),
                retrieved_at="2026-11-04T00:00:00Z",
                origin="official",
            ),
        )
    return manifest


def _results(manifest: Manifest) -> dict[str, str]:
    return {
        "provenance": "official",
        "manifest_digest": manifest.digest(),
        "config_fingerprint": FINGERPRINT,
    }


def test_committed_files_alone_pass(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    for path in (tmp_path / "raw").iterdir():
        path.unlink()
    problems = provenance.verify(
        results=_results(manifest),
        manifest=manifest,
        official_hosts=HOSTS,
        expected_fingerprint=FINGERPRINT,
    )
    assert problems == []


def test_rehash_reports_missing_and_changed_files(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert provenance.rehash(manifest=manifest, data_dir=tmp_path, workers=2, say=print) == []
    (tmp_path / "raw" / "a.zip").write_bytes(b"edited")
    (tmp_path / "raw" / "b.zip").unlink()
    problems = provenance.rehash(manifest=manifest, data_dir=tmp_path, workers=2, say=print)
    assert len(problems) == 2
    assert any("a.zip no longer matches" in p for p in problems)
    assert any("b.zip is missing" in p for p in problems)


def test_off_host_entries_fail(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    entry = manifest.entries["a.zip"]
    manifest.put(
        "a.zip",
        Entry(
            url="https://example.com/a.zip",
            path=entry.path,
            sha256=entry.sha256,
            bytes=entry.bytes,
            retrieved_at=entry.retrieved_at,
            origin="official",
        ),
    )
    problems = provenance.verify(
        results=_results(manifest),
        manifest=manifest,
        official_hosts=HOSTS,
        expected_fingerprint=FINGERPRINT,
    )
    assert problems == ["1 of 2 raw files are not from an official host (for example a.zip)"]
