"""fzt: the factor-zoo-trial command line. Every stage is one command; `results` runs them all."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from factor_zoo_trial import config as config_mod
from factor_zoo_trial.data import build as build_mod
from factor_zoo_trial.data import fetch as fetch_mod
from factor_zoo_trial.data import fixture as fixture_mod
from factor_zoo_trial.data.manifest import Manifest
from factor_zoo_trial.evaluation import pipeline
from factor_zoo_trial.paths import ProjectPaths
from factor_zoo_trial.plotting import figures
from factor_zoo_trial.report import provenance, release, render
from factor_zoo_trial.report import results as results_mod

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Research config (default configs/research.yaml)."),
]
OverrideOpt = Annotated[
    Path | None,
    typer.Option("--override", "-o", help="YAML merged over the config, e.g. configs/ci.yaml."),
]
MirrorOpt = Annotated[
    Path | None, typer.Option("--mirror", help="Copy sources from this folder instead of the web.")
]


def _config(paths: ProjectPaths, config: Path | None, override: Path | None) -> config_mod.Config:
    return config_mod.load(path=config or paths.config, override=override)


def _run(args: list[str], cwd: Path) -> None:
    typer.echo(f"$ {' '.join(args)}")
    completed = subprocess.run(args, cwd=cwd, check=False)  # noqa: S603 - fixed argument lists
    if completed.returncode != 0:
        raise typer.Exit(code=completed.returncode)


@app.command()
def fetch(
    *, config: ConfigOpt = None, override: OverrideOpt = None, mirror: MirrorOpt = None
) -> None:
    """Download every source (resumable, conditional) and update data/manifest.json."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    summary = fetch_mod.fetch_all(config=cfg, paths=paths, mirror=mirror)
    typer.echo(
        f"{summary.files} files, {len(summary.changed)} new or changed; "
        f"{summary.countries} countries and {summary.regions} regions; "
        f"{len(summary.skipped)} locations skipped"
    )


@app.command()
def build(*, config: ConfigOpt = None, override: OverrideOpt = None) -> None:
    """Parse data/raw into Parquet tables in data/processed."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    tables = build_mod.build_tables(config=cfg, paths=paths)
    build_mod.write_tables(tables, paths.processed)
    with_returns = tables.osap["signal"].nunique()
    locations = tables.jkp["location"].nunique()
    factors = tables.jkp["factor"].nunique()
    typer.echo(
        f"{len(tables.signals)} predictors, {with_returns} with returns; "
        f"{locations} JKP locations, {factors} factors"
    )


@app.command()
def estimate(*, config: ConfigOpt = None, override: OverrideOpt = None) -> None:
    """Run every analysis; writes data/runs/estimates.json and reports/tables/*.csv."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    tables = build_mod.read_tables(paths.processed)
    estimates = pipeline.run(config=cfg, tables=tables, tables_dir=paths.tables)
    pipeline.write(estimates, paths.estimates)
    config_mod.write_resolved(cfg, paths.runs / "resolved_config.json")
    typer.echo(f"estimates written in {estimates['runtime_seconds']:.0f} s")


@app.command()
def report(*, config: ConfigOpt = None, override: OverrideOpt = None) -> None:
    """Figures, results.json, README.md and paper/paper.qmd from the estimates."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    estimates = json.loads(paths.estimates.read_text(encoding="utf-8"))
    manifest = Manifest.read(paths.manifest)
    results = results_mod.assemble(config=cfg, estimates=estimates, manifest=manifest)
    results["figures"] = figures.draw_all(
        config=cfg, results=results, tables=paths.tables, out=paths.figures
    )
    results_mod.write(results, paths.results)
    render.render_all(paths, results_mod.read(paths.results))
    typer.echo(f"results.json ({results['provenance']}), README.md and paper/paper.qmd rendered")


@app.command()
def paper() -> None:
    """Build paper/_output/paper/paper.pdf (Typst) and paper.html with Quarto."""
    paths = ProjectPaths.discover()
    quarto = shutil.which("quarto")
    if quarto is None:
        typer.echo("Quarto not found on PATH: install it from https://quarto.org", err=True)
        raise typer.Exit(code=1)
    for fmt in ("typst", "html"):
        _run([quarto, "render", "paper/paper.qmd", "--to", fmt], paths.root)


@app.command()
def results(
    *,
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    mirror: MirrorOpt = None,
    skip_paper: Annotated[bool, typer.Option(help="Do not build the PDF and HTML.")] = False,
) -> None:
    """Everything: fetch, build, estimate, report and (unless skipped) the paper."""
    fetch(config=config, override=override, mirror=mirror)
    build(config=config, override=override)
    estimate(config=config, override=override)
    report(config=config, override=override)
    if not skip_paper:
        paper()


@app.command()
def fixture(
    target: Annotated[Path, typer.Argument(help="Folder to write the synthetic mirror into.")],
    config: ConfigOpt = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
) -> None:
    """Write synthetic sources in the exact upstream formats, for offline runs and CI."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, None)
    fixture_mod.write_fixture(config=cfg, mirror=target, seed=seed)
    typer.echo(f"fixture written to {target}")


@app.command("readme-render")
def readme_render() -> None:
    """Re-render README.md and paper/paper.qmd from the committed results.json."""
    paths = ProjectPaths.discover()
    render.render_all(paths, results_mod.read(paths.results))
    typer.echo("rendered README.md and paper/paper.qmd")


@app.command("check-readme")
def check_readme() -> None:
    """Fail if README.md or paper/paper.qmd differ from a fresh render of results.json."""
    paths = ProjectPaths.discover()
    stale = render.stale(paths, results_mod.read(paths.results))
    if stale:
        typer.echo(f"stale: {', '.join(stale)}; run `uv run fzt readme-render`", err=True)
        raise typer.Exit(code=1)
    typer.echo("README.md and paper/paper.qmd match results.json")


@app.command("check-provenance")
def check_provenance(
    raw: Annotated[
        bool,
        typer.Option(
            help="Re-hash every raw download. --no-raw checks results.json and the "
            "manifest only, for CI, where data/raw is not committed."
        ),
    ] = True,
) -> None:
    """Fail unless results.json was built from the official sources, with intact raw files."""
    paths = ProjectPaths.discover()
    cfg = config_mod.load(path=paths.config)
    problems = provenance.verify(
        results=results_mod.read(paths.results),
        manifest=Manifest.read(paths.manifest),
        raw_dir=paths.raw,
        official_hosts=cfg.sources.official_hosts,
        expected_fingerprint=cfg.fingerprint(),
        require_raw=raw,
    )
    if problems:
        for problem in problems:
            typer.echo(f"provenance: {problem}", err=True)
        raise typer.Exit(code=1)
    if raw:
        typer.echo(
            "results.json comes from the official sources and every raw file matches its hash"
        )
    else:
        typer.echo("results.json comes from the official sources listed in data/manifest.json")


@app.command("check-release")
def check_release(
    write: Annotated[
        bool, typer.Option(help="Pin the newest release in configs/research.yaml.")
    ] = False,
) -> None:
    """Compare the pinned OSAP release with the newest Data-Release branch on Hugging Face."""
    paths = ProjectPaths.discover()
    cfg = config_mod.load(path=paths.config)
    newest = release.newest_release(cfg.sources.osap_api)
    pinned = cfg.sources.osap_revision
    if newest == pinned:
        typer.echo(f"pinned release {pinned} is the newest")
        return
    typer.echo(f"newer release available: {newest} (pinned {pinned})")
    if write:
        release.pin(paths.config, old=pinned, new=newest)
        typer.echo(f"configs/research.yaml now pins {newest}")


@app.command()
def check() -> None:
    """Lint, format, strict types, tests and the README drift check, as CI runs them."""
    paths = ProjectPaths.discover()
    python = sys.executable
    _run([python, "-m", "ruff", "check", "src", "tests"], paths.root)
    _run([python, "-m", "ruff", "format", "--check", "src", "tests"], paths.root)
    _run([python, "-m", "mypy"], paths.root)
    _run([python, "-m", "pytest", "-q"], paths.root)
    check_readme()


@app.command()
def audit() -> None:
    """Known vulnerabilities in the locked runtime dependencies (pip-audit)."""
    paths = ProjectPaths.discover()
    uv = shutil.which("uv")
    if uv is None:
        typer.echo("uv not found on PATH", err=True)
        raise typer.Exit(code=1)
    requirements = paths.runs / "audit-requirements.txt"
    requirements.parent.mkdir(parents=True, exist_ok=True)
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
            "--require-hashes",
            "--requirement",
            str(requirements),
        ],
        paths.root,
    )


if __name__ == "__main__":
    app()
