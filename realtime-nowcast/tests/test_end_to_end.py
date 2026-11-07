"""The whole pipeline on the synthetic fixture: it runs, adds up, recovers the planted yield
response, renders a README that passes the drift check, and is refused by the provenance
guard."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from realtime_nowcast import config as config_mod
from realtime_nowcast.data import build, fetch, fixture
from realtime_nowcast.data.manifest import Manifest
from realtime_nowcast.evaluation import pipeline
from realtime_nowcast.paths import ProjectPaths
from realtime_nowcast.plotting import figures
from realtime_nowcast.report import provenance, render
from realtime_nowcast.report import results as results_mod

ROOT = Path(__file__).resolve().parents[1]
END = "2013-03-31"  # six evaluation quarters, 2011Q3 to 2012Q4
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
    fixture.write_fixture(config=cfg, mirror=mirror, seed=7, end=END)
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


def test_every_quarter_is_scored_against_gdpnow(run: Run) -> None:
    _, results, _, _ = run
    ev = results["estimates"]["evaluation"]
    assert ev["first_quarter"] == "2011Q3"
    assert ev["last_quarter"] == "2012Q4"
    assert ev["quarters"] == 6
    for g in ev["groups"]:
        assert g["all"]["n"] == 6
        assert g["all"]["model_rmse"] > 0
        assert g["all"]["gdpnow_rmse"] > 0


def test_every_change_has_a_cause_and_they_add_up(run: Run) -> None:
    paths, results, _, _ = run
    assert results["estimates"]["history"]["identity_max"] < 1e-9
    changes = pd.read_parquet(paths.runs / "changes.parquet")
    nowcasts = pd.read_parquet(paths.runs / "nowcasts.parquet")
    for quarter, ch in changes.groupby("quarter"):
        entry = ch.loc[ch["kind"] == "entry", "after"].iloc[0]
        total = ch["revisions"].sum() + ch["news"].sum() + ch["reestimation"].sum()
        last = nowcasts.loc[nowcasts["quarter"] == quarter, "nowcast"].iloc[-1]
        assert entry + total == pytest.approx(last, abs=1e-8)


def test_nowcasts_only_use_data_published_by_their_day(run: Run) -> None:
    paths, _, _, _ = run
    releases = pd.read_parquet(paths.runs / "releases.parquet")
    vint = pd.read_parquet(paths.processed / "vintages.parquet")
    first = vint.loc[~vint["value"].isna()].groupby(["series", "period"])["start"].min()
    sample = releases.sample(n=min(300, len(releases)), random_state=0)
    for series, period, day in zip(
        sample["series"].astype(str),
        sample["period"].astype(str),
        pd.to_datetime(sample["date"]),
        strict=True,
    ):
        p = pd.Period(period, freq="M")
        index = p.year * 12 + p.month - 1 - 1970 * 12
        assert first.loc[(series, index)] <= day


def test_a_series_enters_once_alfred_holds_it(run: Run) -> None:
    _, results, _, _ = run
    late = {e["id"]: e["entered"] for e in results["estimates"]["model"]["entered_later"]}
    assert set(late) == set(fixture.LATE_RECORD)
    # ALFRED's first vintage of it comes out mid-April; the next re-estimation takes it in.
    assert late["GACDFSA066MSFRBPHI"] == "2012-05-01"


def test_the_planted_yield_response_is_found(run: Run) -> None:
    _, results, cfg, _ = run
    primary = results["estimates"]["market"][cfg.market.primary]["all"]
    assert primary["slope"] > 0
    assert primary["p"] < 0.01


def test_second_fetch_changes_nothing(run: Run) -> None:
    paths, _, cfg, mirror = run
    summary = fetch.fetch_all(config=cfg, paths=paths, mirror=mirror, say=_quiet)
    assert summary.changed == 0


def test_figures_and_readme(run: Run) -> None:
    paths, results, _, _ = run
    for name in ("rmse", "path", "waterfall", "yields", "errors"):
        assert (paths.figures / f"{name}.png").stat().st_size > 10_000
    readme = paths.readme.read_text(encoding="utf-8")
    assert "synthetic fixture" in readme
    assert "—" not in readme
    assert not render.stale(paths, results)
    paper = paths.paper.read_text(encoding="utf-8")
    assert "<<" not in paper
    assert "—" not in paper


def test_provenance_refuses_fixture_results(run: Run) -> None:
    paths, results, cfg, _ = run
    problems = provenance.verify(
        results=results,
        manifest=Manifest.read(paths.manifest),
        official_hosts=cfg.sources.official_hosts,
        expected_fingerprint=cfg.fingerprint(),
    )
    assert any("synthetic" in p for p in problems)
    assert any("mirror" in p for p in problems)
