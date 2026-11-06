"""The whole pipeline on the synthetic fixture: it runs, recovers the planted truth,
renders a README that passes the drift check, and is refused by the provenance guard."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from factor_zoo_trial import config as config_mod
from factor_zoo_trial.data import build, fetch, fixture
from factor_zoo_trial.data.manifest import Manifest
from factor_zoo_trial.evaluation import pipeline
from factor_zoo_trial.paths import ProjectPaths
from factor_zoo_trial.plotting import figures
from factor_zoo_trial.report import provenance, render
from factor_zoo_trial.report import results as results_mod

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[ProjectPaths, dict[str, Any]]:
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
        config=cfg, tables=build.read_tables(paths.processed), tables_dir=paths.tables
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
    return paths, results_mod.read(paths.results)


def test_decay_recovers_the_planted_truth(run: tuple[ProjectPaths, dict[str, Any]]) -> None:
    _, results = run
    share = results["headline"]["decay"]["share_post_publication"]
    truth = fixture.TRUTH.decay_post_publication
    assert share["low"] - 0.05 <= truth <= share["high"] + 0.05


def test_eb_recovers_the_fixture_hyperparameters(run: tuple[ProjectPaths, dict[str, Any]]) -> None:
    _, results = run
    us = results["headline"]["us"]
    assert us["tau_c"] == pytest.approx(fixture.TRUTH.tau_c, rel=0.5)
    assert us["tau_w"] == pytest.approx(fixture.TRUTH.tau_w, rel=0.5)


def test_space_test_sees_the_shared_alphas(run: tuple[ProjectPaths, dict[str, Any]]) -> None:
    _, results = run
    assert results["headline"]["space"]["share_transfer_beats_no_skill"] > 0.5
    skipped = results["estimates"]["jkp"]["skipped"]
    assert "ken" in skipped
    assert skipped[fixture.EMPTY_COUNTRY] == "factor or market file has no returns"


def test_every_figure_is_written(run: tuple[ProjectPaths, dict[str, Any]]) -> None:
    paths, results = run
    for files in results["figures"].values():
        for relative in files.values():
            assert (paths.root / relative).stat().st_size > 1000


def test_readme_is_current_and_marked_synthetic(run: tuple[ProjectPaths, dict[str, Any]]) -> None:
    paths, results = run
    assert render.stale(paths, results) == []
    assert "synthetic fixture" in paths.readme.read_text()
    paths.readme.write_text(paths.readme.read_text().replace("%", " percent", 1))
    assert render.stale(paths, results) == ["README.md"]
    render.render_all(paths, results)


def test_provenance_guard_refuses_fixture_results(
    run: tuple[ProjectPaths, dict[str, Any]],
) -> None:
    paths, results = run
    cfg = config_mod.load(path=paths.config)
    problems = provenance.verify(
        results=results,
        manifest=Manifest.read(paths.manifest),
        raw_dir=paths.raw,
        official_hosts=cfg.sources.official_hosts,
        expected_fingerprint=cfg.fingerprint(),
    )
    joined = " ".join(problems)
    assert "provenance is 'synthetic'" in joined
    assert "have origin 'mirror'" in joined
    assert "different or overridden configuration" in joined


def test_provenance_guard_catches_a_forged_label_and_a_tampered_file(
    run: tuple[ProjectPaths, dict[str, Any]],
) -> None:
    paths, results = run
    manifest = Manifest.read(paths.manifest)
    forged = {**results, "provenance": "official"}
    entry = next(iter(manifest.entries.values()))
    target = paths.raw / entry.path
    original = target.read_bytes()
    target.write_bytes(original + b"x")
    try:
        problems = provenance.verify(
            results=forged,
            manifest=manifest,
            raw_dir=paths.raw,
            official_hosts=("example.invalid",),
        )
    finally:
        target.write_bytes(original)
    joined = " ".join(problems)
    assert "no longer matches its SHA-256" in joined
    assert "not on an official host" in joined


def test_provenance_without_raw_files_still_refuses_fixture_results(
    run: tuple[ProjectPaths, dict[str, Any]],
) -> None:
    # CI has no data/raw: --no-raw skips the re-hash but must still refuse synthetic
    # results, and must not complain about missing files.
    paths, results = run
    cfg = config_mod.load(path=paths.config)
    problems = provenance.verify(
        results=results,
        manifest=Manifest.read(paths.manifest),
        raw_dir=paths.raw / "absent",
        official_hosts=cfg.sources.official_hosts,
        expected_fingerprint=cfg.fingerprint(),
        require_raw=False,
    )
    joined = " ".join(problems)
    assert "provenance is 'synthetic'" in joined
    assert "is missing" not in joined
