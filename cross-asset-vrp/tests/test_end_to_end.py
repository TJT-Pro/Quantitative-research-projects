"""The whole pipeline on the synthetic fixture: it runs, recovers the planted structure,
renders a README that passes the drift check, and is refused by the provenance guard."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cross_asset_vrp import config as config_mod
from cross_asset_vrp.data import build, fetch, fixture
from cross_asset_vrp.data.manifest import Manifest
from cross_asset_vrp.evaluation import pipeline
from cross_asset_vrp.paths import ProjectPaths
from cross_asset_vrp.plotting import figures
from cross_asset_vrp.report import provenance, render
from cross_asset_vrp.report import results as results_mod

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.e2e
Run = tuple[ProjectPaths, dict[str, Any], config_mod.Config, Path]


def _quiet(_: str) -> None:
    return


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    root = tmp_path_factory.mktemp("project")
    for item in ("pyproject.toml", "README.template.md", "configs"):
        source = ROOT / item
        if source.is_dir():
            shutil.copytree(source, root / item)
        else:
            shutil.copy(source, root / item)
    (root / "paper").mkdir()
    shutil.copy(ROOT / "paper" / "paper.template.qmd", root / "paper" / "paper.template.qmd")
    paths = ProjectPaths(root=root)
    cfg = config_mod.load(path=paths.config, override=root / "configs" / "ci.yaml")
    mirror = root / "mirror"
    fixture.write_fixture(config=cfg, mirror=mirror, seed=7)
    fetch.fetch_all(config=cfg, paths=paths, mirror=mirror, say=_quiet)
    tables = build.build_tables(config=cfg, paths=paths, say=_quiet)
    build.write_tables(tables, paths.processed)
    estimates = pipeline.run(
        config=cfg,
        tables=build.read_tables(paths.processed),
        tables_dir=paths.tables,
        runs_dir=paths.runs,
        say=_quiet,
    )
    pipeline.write(estimates, paths.estimates)
    results = results_mod.assemble(
        config=cfg, estimates=estimates, manifest=Manifest.read(paths.manifest)
    )
    results["figures"] = figures.draw_all(
        config=cfg, results=results, tables=paths.tables, out=paths.figures
    )
    results_mod.write(results, paths.results)
    render.render_all(paths, results_mod.read(paths.results))
    return paths, results_mod.read(paths.results), cfg, mirror


def test_every_market_recovers_its_planted_premium(run: Run) -> None:
    _, results, _, _ = run
    premium = results["estimates"]["premium"]
    for key, planted in fixture.PLANTED.items():
        ratio = premium[key]["ex_post_ratio"]
        assert ratio == pytest.approx(1.0 + planted.premium, abs=0.12), key
        assert premium[key]["ex_post_share_positive"] > 0.5, key


def test_har_beats_persistence_on_the_etfs(run: Run) -> None:
    _, results, cfg, _ = run
    for a in cfg.cboe_assets:
        assert results["estimates"]["premium"][a.key]["har_r2_vs_persistence"] > 0.0, a.key


def test_vx_rescaling_and_coverage(run: Run) -> None:
    paths, results, cfg, _ = run
    vx = pd.read_parquet(paths.processed / "vx.parquet")
    implied = pd.read_parquet(paths.processed / "implied.parquet")
    vix = implied.loc[implied["symbol"] == "VIX"].set_index("date")["value"]
    early = vx.loc[vx["trade_date"] < pd.Timestamp(cfg.futures.rescale_before)]
    ratio = (
        early.set_index("trade_date")["settle"] / vix.reindex(early["trade_date"]).to_numpy()
    ).median()
    assert 0.6 < ratio < 1.6  # the archive's 10x prices were rescaled
    futures = results["estimates"]["futures"]
    assert set(futures["strategies"]) == {"unconditional", "premium", "slope"}
    assert futures["first_trade"] < cfg.futures.rescale_before


def test_bitcoin_noise_kernel_and_replication(run: Run) -> None:
    _, results, _, _ = run
    cal = results["estimates"]["calibration"]
    assert cal["rv1_over_rv5"] > 1.005  # the planted bid-ask bounce shows at one minute
    assert cal["kernel_over_rv5"] == pytest.approx(1.0, abs=0.03)  # and the kernel removes it
    rep = results["estimates"]["replication"]
    assert rep["priced"] == rep["months"] > 0
    assert rep["gap_mean_abs"] < 1.0  # the chain was priced at DVOL: replication returns it
    assert rep["short_mean"] > 0  # DVOL carries the planted premium over realized variance


def test_chains_are_rebuilt_from_the_trade_window(run: Run) -> None:
    paths, results, cfg, _ = run
    chains = pd.read_parquet(paths.processed / "chains.parquet")
    assert len(chains) > 0
    assert (chains["bid"] <= chains["mark"]).all()
    assert (chains["mark"] <= chains["ask"]).all()
    # Every mark comes from a trade inside the window before the snapshot.
    assert chains["age_minutes"].between(0, cfg.replication.window_hours * 60).all()
    assert (chains["expiration"] > chains["snapshot"]).all()
    # The planted 3% half-spread near the money comes back from the trades.
    rep = results["estimates"]["replication"]
    assert rep["atm_half_spread_median"] == pytest.approx(0.03, rel=1e-6)
    # The fixture's marks carry their own volatility; the parity forward must reproduce it.
    assert rep["iv_gap_median_abs"] < 0.01


def test_second_fetch_skips_unchanged_files(run: Run) -> None:
    paths, _, cfg, mirror = run
    summary = fetch.fetch_all(config=cfg, paths=paths, mirror=mirror, say=_quiet)
    manifest = Manifest.read(paths.manifest)
    months = [k for k in manifest.entries if k.startswith("vision/data/")]
    assert summary.btc_months == len(months)
    assert summary.changed == 0


def test_figures_and_readme(run: Run) -> None:
    paths, results, _, _ = run
    assert {
        "premium",
        "calibration",
        "heston",
        "futures",
        "predictability",
        "replication",
        "har",
    } <= set(results["figures"])
    assert render.stale(paths, results) == []
    readme = paths.readme.read_text(encoding="utf-8")
    assert "synthetic fixture" in readme
    paths.readme.write_text(readme.replace("Options imply", "Options suggest"), encoding="utf-8")
    assert render.stale(paths, results) == ["README.md"]
    render.render_all(paths, results)


def test_provenance_refuses_fixture_results(run: Run) -> None:
    paths, results, cfg, _ = run
    problems = provenance.verify(
        results=results,
        manifest=Manifest.read(paths.manifest),
        official_hosts=cfg.sources.official_hosts,
        expected_fingerprint=config_mod.load(path=paths.config).fingerprint(),
    )
    assert any("synthetic" in p for p in problems)
    assert any("mirror" in p for p in problems)
    assert any("configuration" in p for p in problems)
    rehashed = provenance.rehash(
        manifest=Manifest.read(paths.manifest), data_dir=paths.data, workers=2, say=_quiet
    )
    assert rehashed == []
