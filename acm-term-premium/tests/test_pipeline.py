"""End-to-end run of every CLI stage on the synthetic fixture, in an isolated copy.

The run happens in a temporary project directory so it never touches the
committed reports, README or data of the real project.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from acm_term_premium.cli import app
from tests.conftest import ROOT

runner = CliRunner()
OVERRIDES = ["-o", "configs/ci.yaml", "-o", "configs/e2e.yaml"]


def _invoke(*args: str) -> str:
    result = runner.invoke(app, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("project")
    for name in ("pyproject.toml", "README.template.md"):
        shutil.copy(ROOT / name, root / name)
    shutil.copytree(ROOT / "configs", root / "configs")
    (root / "paper").mkdir()
    shutil.copy(ROOT / "paper" / "paper.template.qmd", root / "paper" / "paper.template.qmd")
    # Fewer bootstrap draws keep the run short; every stage still executes.
    (root / "configs" / "e2e.yaml").write_text(
        yaml.safe_dump({"uncertainty": {"replications": 50}, "report": {"figure_dpi": 72}}),
        encoding="utf-8",
    )
    return root


def test_full_pipeline(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    mirror = project / "mirror"
    _invoke("fixture", str(mirror), *OVERRIDES)
    assert "changed: gsw, acm, fedfunds, target, target_upper" in _invoke(
        "fetch", *OVERRIDES, "--mirror", str(mirror)
    )
    _invoke("build", *OVERRIDES)
    _invoke("estimate", *OVERRIDES)
    _invoke("report", *OVERRIDES)

    results = json.loads((project / "reports" / "results.json").read_text(encoding="utf-8"))
    assert results["provenance"] == "synthetic"
    # The fixture workbook is this model's own output, so replication is exact.
    assert results["validation"]["max_abs_all_series_bp"] < 1e-6
    assert results["validation"]["daily_max_abs_bp"] < 1e-6
    assert results["validation"]["convention_gap_max_abs_bp"] > 0.01
    for paths in results["figures"].values():
        for path in paths:
            assert (project / path).is_file()
    readme = (project / "README.md").read_text(encoding="utf-8")
    assert "Synthetic data" in readme
    for text in (readme, (project / "paper" / "paper.qmd").read_text(encoding="utf-8")):
        assert "{{" not in text
        assert "{%" not in text
        assert chr(0x2014) not in text, "no em dashes in rendered prose"

    _invoke("check-readme")
    failed = runner.invoke(app, ["check-provenance"])
    assert failed.exit_code == 1, "fixture results must never pass the publish guard"

    (project / "README.md").write_text(readme + "edited by hand\n", encoding="utf-8")
    assert runner.invoke(app, ["check-readme"]).exit_code == 1
    _invoke("readme-render")
    _invoke("check-readme")


def test_estimate_refuses_a_missing_raw_file(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(project)
    victim = project / "data" / "raw" / "DFEDTAR.csv"
    saved = victim.read_bytes()
    victim.unlink()
    try:
        result = runner.invoke(app, ["build", *OVERRIDES])
        assert result.exit_code != 0
        assert "DFEDTAR.csv" in str(result.exception)
    finally:
        victim.write_bytes(saved)
