"""Period labels respect knowledge time; every parser reads the fixture's exact formats."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from factor_zoo_trial import config as config_mod
from factor_zoo_trial.data import parsers, sources
from factor_zoo_trial.data.download import mirror_path
from factor_zoo_trial.data.fixture import write_fixture
from factor_zoo_trial.evaluation import osap_panel
from factor_zoo_trial.report import release

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> config_mod.Config:
    return config_mod.load(path=ROOT / "configs" / "research.yaml")


@pytest.fixture(scope="module")
def mirror(tmp_path_factory: pytest.TempPathFactory, cfg: config_mod.Config) -> Path:
    target = tmp_path_factory.mktemp("mirror")
    write_fixture(config=cfg, mirror=target, seed=3)
    return target


@given(
    start=st.integers(1930, 2000),
    length=st.integers(1, 40),
    gap=st.integers(0, 8),
    shift=st.integers(0, 5),
    month=st.integers(1, 12),
)
def test_no_month_is_labelled_post_publication_before_it_could_be_known(
    start: int, length: int, gap: int, shift: int, month: int
) -> None:
    end = start + length
    year = end + gap
    months = pd.date_range(f"{start - 3}-01-01", f"{year + 5}-12-01", freq="MS")
    returns = pd.DataFrame({"signal": "s", "month": months, "ret": 0.0})
    signals = pd.DataFrame(
        {"signal": ["s"], "year": [year], "sample_start": [start], "sample_end": [end]}
    )
    labelled = osap_panel.label_periods(returns, signals, publication_month=month, shift=shift)
    published = pd.Timestamp(year=year - shift, month=month, day=1)
    post = labelled.loc[labelled["period"] == "post_publication", "month"]
    inside = labelled.loc[labelled["period"] == "in_sample", "month"]
    assert (post > published).all()
    assert inside.dt.year.between(start, end).all()
    assert labelled["month"].dt.year.min() >= start  # pre-sample months are dropped


def test_in_sample_tests_ignore_later_returns() -> None:
    months = pd.date_range("1970-01-01", "2010-12-01", freq="MS")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        {"signal": "s", "month": months, "ret": rng.standard_normal(months.size)}
    )
    signals = pd.DataFrame(
        {"signal": ["s"], "year": [1995], "sample_start": [1970], "sample_end": [1990]}
    )
    base = osap_panel.label_periods(returns, signals, publication_month=12, shift=0)
    future = returns.assign(ret=np.where(months.year > 1990, 99.0, returns["ret"]))
    moved = osap_panel.label_periods(future, signals, publication_month=12, shift=0)
    a = osap_panel.in_sample_tests(base, coefficient=0.4, min_months=36)
    b = osap_panel.in_sample_tests(moved, coefficient=0.4, min_months=36)
    pd.testing.assert_frame_equal(a, b)


def test_signal_doc_reads_and_drops_placebos(cfg: config_mod.Config, mirror: Path) -> None:
    doc = parsers.read_signal_doc(
        mirror_path(mirror, sources.osap_target(cfg, cfg.osap.signal_doc).url)
    )
    assert {"Predictor", "Placebo"} <= set(doc["kind"])
    assert doc["year"].notna().all()


def test_signal_doc_missing_column_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "SignalDoc.csv"
    path.write_text("Acronym,Year\nX,2000\n")
    with pytest.raises(parsers.SchemaError, match="missing columns"):
        parsers.read_signal_doc(path)


def test_long_short_reads_only_ls_rows(cfg: config_mod.Config, mirror: Path) -> None:
    path = mirror_path(mirror, sources.osap_target(cfg, cfg.osap.variants["original"]).url)
    ls = parsers.read_long_short(path, port="LS")
    assert list(ls.columns) == ["signal", "month", "ret"]
    assert (ls["month"].dt.day == 1).all()


def test_jkp_zip_and_market(cfg: config_mod.Config, mirror: Path) -> None:
    factors = parsers.read_jkp_factors(
        mirror_path(
            mirror, sources.jkp_location_target(cfg, location="usa", selector=cfg.jkp.selector).url
        )
    )
    market = parsers.read_jkp_market(
        mirror_path(
            mirror,
            sources.jkp_location_target(cfg, location="usa", selector=cfg.jkp.market_selector).url,
        )
    )
    assert factors["ret"].abs().median() > 0.5  # converted to percent
    assert market["month"].is_unique


def test_jkp_regional_files_count_countries(cfg: config_mod.Config, mirror: Path) -> None:
    region = parsers.read_jkp_factors(
        mirror_path(
            mirror,
            sources.jkp_location_target(cfg, location="developed", selector=cfg.jkp.selector).url,
        )
    )
    country = parsers.read_jkp_factors(
        mirror_path(
            mirror, sources.jkp_location_target(cfg, location="gbr", selector=cfg.jkp.selector).url
        )
    )
    assert region["aggregate"].all()
    assert region["n_countries"].notna().all()
    assert not country["aggregate"].any()


def test_jkp_file_without_any_count_fails_loudly(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"location": ["usa"], "name": ["be_me"], "date": ["2020-01-31"], "ret": [0.01]}
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("x.csv", frame.to_csv(index=False))
    (tmp_path / "x.zip").write_bytes(buffer.getvalue())
    with pytest.raises(parsers.SchemaError, match="n_countries"):
        parsers.read_jkp_factors(tmp_path / "x.zip")


def test_jkp_eom_column_is_accepted(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "location": ["usa"],
            "name": ["be_me"],
            "eom": ["2020-01-31"],
            "ret": [0.01],
            "n_stocks_min": [30],
        }
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("x.csv", frame.to_csv(index=False))
    (tmp_path / "x.zip").write_bytes(buffer.getvalue())
    out = parsers.read_jkp_factors(tmp_path / "x.zip")
    assert out["month"].iloc[0] == pd.Timestamp("2020-01-01")
    assert out["ret"].iloc[0] == pytest.approx(1.0)


def test_french_parser_stops_before_the_annual_block(cfg: config_mod.Config, mirror: Path) -> None:
    path = mirror_path(mirror, f"{cfg.sources.french_base}/{cfg.french.five_factor_zip}")
    frame = parsers.read_french_monthly(path)
    assert list(frame.columns) == ["month", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert frame["month"].is_monotonic_increasing
    assert frame["month"].dt.year.min() == 1963


def test_location_plan_skips_files_that_are_not_offered(
    cfg: config_mod.Config, mirror: Path
) -> None:
    plan = sources.plan_locations(
        cfg,
        availability_path=mirror_path(mirror, sources.jkp_availability_target(cfg).url),
        country_codes={"usa", "gbr", "nzl"},
    )
    assert "nzl" in plan.skipped
    assert "usa" in plan.regions
    assert "usa" not in plan.countries


def test_release_ordering() -> None:
    names = ["main", "2025.10", "Data-Release-2026.10", "2024.08", "Test-Data-2026.10"]
    assert release.newest_from(names) == "Data-Release-2026.10"


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    text = (ROOT / "configs" / "research.yaml").read_text() + "\nsurprise: 1\n"
    (tmp_path / "bad.yaml").write_text(text)
    with pytest.raises(ValueError, match="surprise"):
        config_mod.load(path=tmp_path / "bad.yaml")


def test_override_changes_the_fingerprint(cfg: config_mod.Config) -> None:
    ci = config_mod.load(
        path=ROOT / "configs" / "research.yaml", override=ROOT / "configs" / "ci.yaml"
    )
    assert ci.fingerprint() != cfg.fingerprint()
    assert (
        cfg.fingerprint() == config_mod.load(path=ROOT / "configs" / "research.yaml").fingerprint()
    )
