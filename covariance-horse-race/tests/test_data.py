from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from covariance_horse_race.config import ResearchConfig, load_config
from covariance_horse_race.data import french, store
from covariance_horse_race.data.fixture import load_fixture_spec, write_fixture
from covariance_horse_race.data.source import (
    SourceError,
    check_files_listed,
    fetch_all,
    read_single_csv,
    verify_raw,
    zip_links,
)
from tests.conftest import ROOT

PORTFOLIO_TEXT = """This file was created by CMPT_IND_RETS_DAILY using the 202509 CRSP database.
It contains value- and equal-weighted returns for 3 industry portfolios.
Missing data are indicated by -99.99 or -999.

  Average Value Weighted Returns -- Daily
         ,NoDur ,Durbl ,Manuf ,
19260701,   0.10,  -0.25, -99.99,
19260702,   0.45,   0.10,   0.20,

  Average Equal Weighted Returns -- Daily
         ,NoDur ,Durbl ,Manuf
19260701,   1.10,   1.25,   1.30
19260702,   1.45,   1.10,   1.20

 Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

FACTOR_TEXT = """This file was created by CMPT_ME_BEME_RETS_DAILY.
,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.25,   -0.27,    0.01
19260702,    0.45,   -0.33,   -0.06,    0.01
"""


def test_value_weighted_section_selected_and_scaled() -> None:
    frame = french.parse(PORTFOLIO_TEXT.encode(), "value_weighted", [-99.99, -999.0], 100.0)
    assert list(frame.columns) == ["NoDur", "Durbl", "Manuf"]
    assert frame.index[0] == pd.Timestamp("1926-07-01")
    assert frame.loc["1926-07-01", "NoDur"] == pytest.approx(0.001)
    assert np.isnan(frame.loc["1926-07-01", "Manuf"])


def test_equal_weighted_section_selected() -> None:
    frame = french.parse(PORTFOLIO_TEXT.encode(), "equal_weighted", [-99.99], 100.0)
    assert frame.loc["1926-07-02", "Manuf"] == pytest.approx(0.012)


def test_factor_file_single_block() -> None:
    frame = french.parse(FACTOR_TEXT.encode(), None, [-99.99], 100.0)
    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RF"]


def test_crlf_and_latin1_are_handled() -> None:
    payload = (
        PORTFOLIO_TEXT.replace("\n", "\r\n")
        .replace("Copyright", "Copyright \xa9")
        .encode("latin-1")
    )
    frame = french.parse(payload, "value_weighted", [-99.99], 100.0)
    assert frame.shape == (2, 3)


def test_ambiguous_factor_block_rejected() -> None:
    with pytest.raises(SourceError, match="expected one block"):
        french.parse(PORTFOLIO_TEXT.encode(), None, [-99.99], 100.0)


def test_ragged_row_rejected() -> None:
    bad = FACTOR_TEXT.replace("0.45,   -0.33,   -0.06,    0.01", "0.45,   -0.33")
    with pytest.raises(SourceError, match="values for"):
        french.parse(bad.encode(), None, [-99.99], 100.0)


def test_non_monotonic_dates_rejected() -> None:
    bad = FACTOR_TEXT.replace("19260702", "19260630")
    with pytest.raises(SourceError, match="strictly increasing"):
        french.parse(bad.encode(), None, [-99.99], 100.0)


def test_zip_links_resolves_relative_hrefs() -> None:
    html = (
        '<a href="ftp/A_daily_CSV.zip">CSV</a> <A HREF="other.html">x</A> <a href="ftp/B.zip">B</a>'
    )
    links = zip_links(html, "https://example.org/pages/library.html")
    assert links == {
        "A_daily_CSV.zip": "https://example.org/pages/ftp/A_daily_CSV.zip",
        "B.zip": "https://example.org/pages/ftp/B.zip",
    }


def test_missing_link_names_close_matches(config: ResearchConfig) -> None:
    links = {
        name.replace("daily", "Daily"): "https://x/" + name for name in config.data.files.values()
    }
    with pytest.raises(SourceError, match="closest links"):
        check_files_listed(config.data, links)


def _zip(members: dict[str, bytes], compress: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compress) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ({"a.csv": b"1", "b.csv": b"2"}, "expected one CSV"),
        ({"readme.txt": b"1"}, "expected one CSV"),
        ({"../evil.csv": b"1"}, "unsafe member path"),
        ({"bomb.csv": b"0" * 5_000_000}, "bomb"),
    ],
)
def test_unsafe_archives_rejected(tmp_path: Path, members: dict[str, bytes], message: str) -> None:
    path = tmp_path / "x.zip"
    path.write_bytes(_zip(members))
    with pytest.raises(SourceError, match=message):
        read_single_csv(path)


def test_corrupt_archive_rejected(tmp_path: Path) -> None:
    path = tmp_path / "x.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(SourceError, match="not a valid zip"):
        read_single_csv(path)


@pytest.fixture(scope="module")
def mirror(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ResearchConfig]:
    cfg = load_config(ROOT / "configs" / "research.yaml", [ROOT / "configs" / "ci.yaml"])
    target = tmp_path_factory.mktemp("mirror")
    write_fixture(cfg, load_fixture_spec(ROOT / "configs" / "fixture.yaml"), target)
    return target, cfg


def test_fetch_records_manifest_and_detects_changes(
    tmp_path: Path, mirror: tuple[Path, ResearchConfig]
) -> None:
    source, cfg = mirror
    raw, manifest = tmp_path / "raw", tmp_path / "manifest.json"
    raw.mkdir()
    first = fetch_all(cfg.data, raw, manifest, mirror=source)
    assert sorted(first.changed) == sorted(cfg.data.files)
    second = fetch_all(cfg.data, raw, manifest, mirror=source)
    assert second.changed == []
    recorded = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    assert set(recorded) == set(cfg.data.files)
    verify_raw(raw, manifest)

    tampered = raw / cfg.data.files["ind10"]
    tampered.write_bytes(tampered.read_bytes() + b"\x00")
    with pytest.raises(SourceError, match="manifest hash"):
        verify_raw(raw, manifest)


def test_build_and_factor_join(tmp_path: Path, mirror: tuple[Path, ResearchConfig]) -> None:
    source, cfg = mirror
    raw, manifest, processed = tmp_path / "raw", tmp_path / "manifest.json", tmp_path / "proc"
    raw.mkdir()
    fetch_all(cfg.data, raw, manifest, mirror=source)
    store.build_processed(cfg, raw, processed, manifest)
    ind10 = store.load(processed, "ind10")
    assert ind10.shape[1] == 10
    assert ind10.isna().any().any(), "fixture marks late-starting portfolios with the sentinel"
    extended = store.load_factor_set(processed, cfg.factors.extended)
    primary = store.load_factor_set(processed, cfg.factors.primary)
    assert list(extended.columns) == cfg.factors.extended.columns
    assert extended.index[0] > primary.index[0], "five-factor file starts later"
    assert not extended.isna().any().any()
    rows = store.coverage(processed, ["ind10", "ff3"])
    assert rows[0].n_columns == 10
    assert rows[0].missing_share > 0
