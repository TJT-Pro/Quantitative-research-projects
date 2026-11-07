"""The whole pipeline on the synthetic fixture: it runs, recovers the planted structure,
renders a README that passes the drift check, and is refused by the provenance guard."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from perp_funding_carry import config as config_mod
from perp_funding_carry.data import books, build, fetch, fixture
from perp_funding_carry.data.manifest import Manifest
from perp_funding_carry.evaluation import pipeline
from perp_funding_carry.paths import ProjectPaths
from perp_funding_carry.plotting import figures
from perp_funding_carry.report import provenance, render
from perp_funding_carry.report import results as results_mod

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ProjectPaths, dict[str, Any], config_mod.Config]:
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
    fetch.fetch_all(config=cfg, paths=paths, mirror=mirror, log=False)
    tables = build.build_tables(config=cfg, paths=paths)
    build.write_tables(tables, paths.processed)
    estimates = pipeline.run(
        config=cfg,
        tables=build.read_tables(paths.processed),
        snapshots=books.read_store(paths.data / books.STORE),
        tables_dir=paths.tables,
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
    return paths, results_mod.read(paths.results), cfg


def test_universe_filters_and_maps_contracts(
    run: tuple[ProjectPaths, dict[str, Any], config_mod.Config],
) -> None:
    paths, results, _ = run
    contracts = pd.read_parquet(paths.processed / "contracts.parquet").set_index("symbol")
    assert "BTCUSDC" not in contracts.index  # not USDT-quoted
    assert "NOSPOTUSDT" not in contracts.index  # no spot pair
    assert contracts.loc["1000PEPEUSDT", "spot"] == "PEPEUSDT"
    assert bool(contracts.loc["OLDUSDT", "delisted"])
    assert results["estimates"]["universe"]["delisted"] == 1


def test_rebuild_recovers_the_planted_convention(
    run: tuple[ProjectPaths, dict[str, Any], config_mod.Config],
) -> None:
    _, results, _ = run
    rb = results["estimates"]["rebuild"]
    assert rb["best"] == "per_interval"
    assert rb["conventions"]["per_interval"]["share_within"] == pytest.approx(1.0)
    # The FAQ reading differs only for the contract that moved to 4-hour funding.
    assert rb["conventions"]["faq"]["by_interval"]["8"]["share"] == pytest.approx(1.0)
    assert rb["conventions"]["faq"]["by_interval"]["4"]["share"] < 0.9


def test_spike_liquidates_without_top_ups(
    run: tuple[ProjectPaths, dict[str, Any], config_mod.Config],
) -> None:
    _, results, _ = run
    paths = results["headline"]["paths"]
    assert paths["liquidate"]["liquidations"] >= 1
    assert paths["ideal"]["liquidations"] == 0
    assert results["headline"]["portfolio"]["delisted_closes"] >= 1


def test_okx_history_lives_in_the_local_store(
    run: tuple[ProjectPaths, dict[str, Any], config_mod.Config],
) -> None:
    paths, results, cfg = run
    store = pd.read_csv(paths.collected / "okx_funding.csv")
    assert set(store["origin"]) == {"mirror"}
    venues = results["estimates"]["venues"][cfg.universe.core_symbols[0]]
    assert set(venues["all"]["annualized"]) == {"binance", "bybit", "okx"}


def test_a_second_fetch_skips_every_unchanged_file(
    run: tuple[ProjectPaths, dict[str, Any], config_mod.Config],
) -> None:
    paths, _, cfg = run
    summary = fetch.fetch_all(config=cfg, paths=paths, mirror=paths.root / "mirror", log=False)
    assert summary.requested == 0
    assert summary.okx_rows_added == 0


def test_every_figure_is_written(
    run: tuple[ProjectPaths, dict[str, Any], config_mod.Config],
) -> None:
    paths, results, _ = run
    assert {"headline", "heatmap", "cross_section", "forecast", "drawdown"} <= set(
        results["figures"]
    )
    for files in results["figures"].values():
        for relative in files.values():
            assert (paths.root / relative).stat().st_size > 1000


def test_readme_is_current_and_marked_synthetic(
    run: tuple[ProjectPaths, dict[str, Any], config_mod.Config],
) -> None:
    paths, results, _ = run
    assert render.stale(paths, results) == []
    assert "synthetic fixture" in paths.readme.read_text()
    paths.readme.write_text(paths.readme.read_text().replace("%", " percent", 1))
    assert render.stale(paths, results) == ["README.md"]
    render.render_all(paths, results)


def test_provenance_guard_refuses_fixture_results(
    run: tuple[ProjectPaths, dict[str, Any], config_mod.Config],
) -> None:
    paths, results, _ = run
    cfg = config_mod.load(path=paths.config)
    problems = " ".join(
        provenance.verify(
            results=results,
            manifest=Manifest.read(paths.manifest),
            official_hosts=cfg.sources.official_hosts,
            expected_fingerprint=cfg.fingerprint(),
        )
    )
    assert "provenance is 'synthetic'" in problems
    assert "have origin 'mirror'" in problems
    assert "different or overridden configuration" in problems


def test_spread_model_recovers_the_planted_law(
    run: tuple[ProjectPaths, dict[str, Any], config_mod.Config],
) -> None:
    _, results, _ = run
    model = results["estimates"]["spread_model"]
    assert model["contracts"] == 10  # every live contract; the delisted one has no book
    assert model["perp"]["slope"] == pytest.approx(-fixture.SPREAD_POWER, abs=0.15)
    assert model["spot"]["slope"] == pytest.approx(-fixture.SPREAD_POWER, abs=0.15)


def test_costs_and_their_decomposition(
    run: tuple[ProjectPaths, dict[str, Any], config_mod.Config],
) -> None:
    _, results, _ = run
    pf = results["estimates"]["portfolio"]
    by_multiple = {row["cost_multiple"]: row["annual_return"] for row in pf["cost_sensitivity"]}
    ordered = [by_multiple[m] for m in sorted(by_multiple)]
    assert ordered == sorted(ordered, reverse=True)  # higher costs never help
    d = pf["decomposition"]
    assert d["fees"] < 0
    assert d["spread"] < 0
    assert d["funding"] + d["fees"] + d["spread"] + d["other"] == pytest.approx(d["total"])
