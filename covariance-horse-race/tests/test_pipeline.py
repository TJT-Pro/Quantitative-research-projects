"""End-to-end run of every CLI stage on a synthetic mirror, in an isolated copy.

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

from covariance_horse_race.cli import app
from tests.conftest import ROOT

runner = CliRunner()


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
    # A shorter fixture and a lighter workload keep the full run to seconds.
    spec = yaml.safe_load((ROOT / "configs" / "fixture.yaml").read_text(encoding="utf-8"))
    spec["n_days"] = 800
    (root / "configs" / "fixture.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    (root / "configs" / "e2e.yaml").write_text(
        yaml.safe_dump(
            {
                "universes": [
                    {"key": "ind10", "file": "ind10", "label": "10 industries"},
                    {"key": "ind30", "file": "ind30", "label": "30 industries"},
                ],
                "report": {"headline_universe": "ind10", "spectrum_universe": "ind10"},
                "evaluation": {"mcs": {"bootstrap_replications": 200, "min_months": 12}},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_full_pipeline(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    overrides = ["-o", "configs/ci.yaml", "-o", "configs/e2e.yaml"]
    mirror = project / "mirror"
    _invoke("fixture", str(mirror), *overrides)
    assert "changed since last manifest" in _invoke("fetch", *overrides, "--mirror", str(mirror))
    assert "changed since last manifest: none" in _invoke(
        "fetch", *overrides, "--mirror", str(mirror)
    )
    _invoke("build", *overrides)
    # Two universes on two spawned workers: the process model Windows uses.
    _invoke("backtest", *overrides, "--workers", "2")
    _invoke("simulate", *overrides)
    _invoke("report", *overrides)

    results = json.loads((project / "reports" / "results.json").read_text(encoding="utf-8"))
    assert results["provenance"] == "synthetic"
    assert results["headline"]["universe"] == "ind10"
    for paths in results["figures"].values():
        for path in paths:
            assert (project / path).is_file()
    readme = (project / "README.md").read_text(encoding="utf-8")
    assert "Synthetic data" in readme
    assert "{{" not in readme
    assert chr(0x2014) not in readme, "no em dashes in rendered prose"

    _invoke("check-readme")
    failed = runner.invoke(app, ["check-provenance"])
    assert failed.exit_code == 1, "fixture results must never pass the publish guard"

    (project / "README.md").write_text(readme + "edited by hand\n", encoding="utf-8")
    stale = runner.invoke(app, ["check-readme"])
    assert stale.exit_code == 1
    _invoke("readme-render")
    _invoke("check-readme")


def test_report_requires_matching_backtest(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A config whose numbers differ (new fingerprint) has no backtest yet and must not report."""
    monkeypatch.chdir(project)
    (project / "configs" / "alpha.yaml").write_text(
        yaml.safe_dump({"evaluation": {"mcs": {"alpha": 0.05}}}), encoding="utf-8"
    )
    overrides = ["-o", "configs/ci.yaml", "-o", "configs/e2e.yaml", "-o", "configs/alpha.yaml"]
    with pytest.raises(FileNotFoundError, match="no backtest"):
        runner.invoke(app, ["report", *overrides], catch_exceptions=False)
