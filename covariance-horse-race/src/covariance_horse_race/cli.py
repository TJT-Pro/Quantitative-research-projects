"""Command line entry points: `chr <stage>`."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from covariance_horse_race import pipeline
from covariance_horse_race.config import ResearchConfig, load_config
from covariance_horse_race.data.fixture import load_fixture_spec, write_fixture
from covariance_horse_race.paths import ProjectPaths
from covariance_horse_race.report import readme
from covariance_horse_race.report.results import read_results

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Covariance forecasting horse race on the Kenneth French daily data.",
)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Base research config.")]
OverrideOpt = Annotated[
    list[Path] | None, typer.Option("--override", "-o", help="Override files, applied in order.")
]


def _load(config: Path, override: list[Path] | None) -> tuple[ResearchConfig, ProjectPaths]:
    paths = ProjectPaths.discover()
    return load_config(
        paths.root / config if not config.is_absolute() else config,
        [paths.root / o if not o.is_absolute() else o for o in override or []],
    ), paths


_DEFAULT_CONFIG = Path("configs/research.yaml")


@app.command()
def fetch(
    config: ConfigOpt = _DEFAULT_CONFIG,
    override: OverrideOpt = None,
    mirror: Annotated[
        Path | None, typer.Option(help="Local mirror instead of the network.")
    ] = None,
) -> None:
    """Download every configured file, verify it and update data/manifest.json."""
    cfg, paths = _load(config, override)
    result = pipeline.fetch(cfg, paths, mirror, typer.echo)
    changed = ", ".join(result.changed) if result.changed else "none"
    typer.echo(f"fetched {len(result.entries)} files; changed since last manifest: {changed}")
    (paths.root / "data" / "changed.txt").write_text(
        "\n".join(result.changed) + "\n", encoding="utf-8"
    )


@app.command()
def build(config: ConfigOpt = _DEFAULT_CONFIG, override: OverrideOpt = None) -> None:
    """Parse raw zips into Parquet after checking them against the manifest."""
    cfg, paths = _load(config, override)
    pipeline.build(cfg, paths)
    typer.echo(f"built processed datasets in {paths.processed.relative_to(paths.root)}")


@app.command()
def backtest(
    config: ConfigOpt = _DEFAULT_CONFIG,
    override: OverrideOpt = None,
    workers: Annotated[int | None, typer.Option(help="Parallel universes.")] = None,
) -> None:
    """Run the rolling out-of-sample backtest for every universe, window and model."""
    cfg, paths = _load(config, override)
    pipeline.backtest(cfg, paths, workers, typer.echo)


@app.command()
def simulate(config: ConfigOpt = _DEFAULT_CONFIG, override: OverrideOpt = None) -> None:
    """Run the known-covariance simulation study."""
    cfg, paths = _load(config, override)
    pipeline.simulate(cfg, paths, typer.echo)


@app.command()
def report(config: ConfigOpt = _DEFAULT_CONFIG, override: OverrideOpt = None) -> None:
    """Build tables, figures, results.json and README.md from the latest run."""
    cfg, paths = _load(config, override)
    pipeline.report(cfg, paths, typer.echo)


@app.command()
def readme_render() -> None:
    """Re-render README.md and paper/paper.qmd from the committed results.json."""
    paths = ProjectPaths.discover()
    results = read_results(paths.results)
    readme.write(paths.readme_template, results, paths.readme)
    readme.write(paths.paper_template, results, paths.paper)
    typer.echo("rendered README.md and paper/paper.qmd")


@app.command("check-readme")
def check_readme() -> None:
    """Fail if README.md or the paper differ from a fresh render of results.json."""
    paths = ProjectPaths.discover()
    results = read_results(paths.results)
    stale = [
        output.relative_to(paths.root).as_posix()
        for template, output in (
            (paths.readme_template, paths.readme),
            (paths.paper_template, paths.paper),
        )
        if not readme.is_current(template, results, output)
    ]
    if stale:
        typer.echo(f"stale: {', '.join(stale)}; run `make readme`", err=True)
        raise typer.Exit(code=1)
    typer.echo("README.md and paper/paper.qmd match results.json")


@app.command("check-provenance")
def check_provenance() -> None:
    """Fail unless results.json was produced from the library data, not the fixture."""
    paths = ProjectPaths.discover()
    source = read_results(paths.results).get("provenance")
    if source != "library":
        typer.echo(f"results.json provenance is {source!r}; publish only library results", err=True)
        raise typer.Exit(code=1)
    typer.echo("results.json comes from the library data")


@app.command()
def fixture(
    target: Annotated[Path, typer.Argument(help="Directory to write the mirror into.")],
    config: ConfigOpt = _DEFAULT_CONFIG,
    override: OverrideOpt = None,
    spec: Annotated[Path, typer.Option(help="Fixture shape.")] = Path("configs/fixture.yaml"),
) -> None:
    """Write a synthetic mirror of the library files for offline runs."""
    cfg, paths = _load(config, override)
    write_fixture(cfg, load_fixture_spec(paths.root / spec), target)
    typer.echo(f"wrote fixture mirror to {target}")


def _run(args: list[str], cwd: Path) -> None:
    """Run one tool with fixed arguments; stop on the first failure."""
    typer.echo(f"> {' '.join(args)}")
    result = subprocess.run(args, cwd=cwd, check=False)  # noqa: S603 - fixed argument lists
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


@app.command()
def paper() -> None:
    """Build paper/_output/paper/paper.pdf and paper.html with Quarto."""
    paths = ProjectPaths.discover()
    quarto = shutil.which("quarto")
    if quarto is None:
        typer.echo("Quarto not found on PATH: install it from https://quarto.org", err=True)
        raise typer.Exit(code=1)
    for fmt in ("typst", "html"):
        _run([quarto, "render", "paper/paper.qmd", "--to", fmt], paths.root)


@app.command()
def results(
    config: ConfigOpt = _DEFAULT_CONFIG,
    override: OverrideOpt = None,
    workers: Annotated[int | None, typer.Option(help="Parallel universes.")] = None,
    skip_paper: Annotated[bool, typer.Option(help="Do not build the PDF.")] = False,
) -> None:
    """Everything, from download to README and paper."""
    cfg, paths = _load(config, override)
    report_fetch = pipeline.fetch(cfg, paths, None, typer.echo)
    typer.echo(f"fetched {len(report_fetch.entries)} files; changed: {report_fetch.changed}")
    pipeline.build(cfg, paths)
    pipeline.backtest(cfg, paths, workers, typer.echo)
    pipeline.simulate(cfg, paths, typer.echo)
    pipeline.report(cfg, paths, typer.echo)
    if not skip_paper:
        paper()


@app.command()
def check() -> None:
    """Lint, formatting, strict types, tests, and README drift."""
    paths = ProjectPaths.discover()
    python = sys.executable
    _run([python, "-m", "ruff", "check", "src", "tests"], paths.root)
    _run([python, "-m", "ruff", "format", "--check", "src", "tests"], paths.root)
    _run([python, "-m", "mypy"], paths.root)
    _run([python, "-m", "pytest"], paths.root)
    check_readme()


@app.command()
def audit() -> None:
    """Known vulnerabilities in the locked runtime dependencies."""
    paths = ProjectPaths.discover()
    uv = shutil.which("uv")
    if uv is None:
        typer.echo("uv not found on PATH", err=True)
        raise typer.Exit(code=1)
    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements.txt"
        _run(
            [
                uv,
                "export",
                "--frozen",
                "--no-dev",
                "--no-emit-project",
                "--format",
                "requirements-txt",
                "--output-file",
                str(requirements),
            ],
            paths.root,
        )
        _run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--strict",
                "--disable-pip",
                "--requirement",
                str(requirements),
            ],
            paths.root,
        )


if __name__ == "__main__":
    app()
