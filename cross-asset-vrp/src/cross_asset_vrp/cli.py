"""vrp: the cross-asset-vrp command line. Every stage is one command; `results` runs them all."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from cross_asset_vrp import config as config_mod
from cross_asset_vrp.data import build as build_mod
from cross_asset_vrp.data import fetch as fetch_mod
from cross_asset_vrp.data import fixture as fixture_mod
from cross_asset_vrp.data.manifest import Manifest
from cross_asset_vrp.evaluation import pipeline
from cross_asset_vrp.paths import ProjectPaths
from cross_asset_vrp.plotting import figures
from cross_asset_vrp.report import provenance, render
from cross_asset_vrp.report import results as results_mod

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
    Path | None,
    typer.Option("--mirror", help="Read every source from this folder instead of the web."),
]


def _say(line: str) -> None:
    typer.echo(line)


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
    """Download what changed from Cboe, Tiingo, FRED, Binance and Deribit."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    s = fetch_mod.fetch_all(config=cfg, paths=paths, mirror=mirror, say=_say)
    typer.echo(
        f"{s.files:,} files ({s.requested:,} requested, {s.changed:,} new or changed); "
        f"{len(s.missing_contracts)} VX months never listed; {s.btc_months} bitcoin months; "
        f"{s.snapshots} months of option trades"
    )


@app.command()
def build(*, config: ConfigOpt = None, override: OverrideOpt = None) -> None:
    """Parse data/raw into Parquet tables in data/processed."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    tables = build_mod.build_tables(config=cfg, paths=paths, say=_say)
    build_mod.write_tables(tables, paths.processed)
    typer.echo(
        f"{tables.implied['symbol'].nunique()} implied indices, "
        f"{tables.etf['ticker'].nunique()} ETFs, "
        f"{tables.vx['expiry'].nunique()} VX contracts, {len(tables.btc_daily):,} bitcoin days, "
        f"{tables.chains['snapshot'].nunique() if len(tables.chains) else 0} chain snapshots"
    )


@app.command()
def estimate(*, config: ConfigOpt = None, override: OverrideOpt = None) -> None:
    """Run every analysis; writes data/runs/estimates.json and reports/tables/*.csv."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    tables = build_mod.read_tables(paths.processed)
    estimates = pipeline.run(
        config=cfg, tables=tables, tables_dir=paths.tables, runs_dir=paths.runs, say=_say
    )
    pipeline.write(estimates, paths.estimates)
    config_mod.write_resolved(cfg, paths.runs / "resolved_config.json")
    typer.echo(f"estimates written in {estimates['runtime_seconds']:.0f} s")


@app.command()
def report(*, config: ConfigOpt = None, override: OverrideOpt = None) -> None:
    """Figures, results.json, README.md and paper/paper.qmd from the estimates."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    estimates = json.loads(paths.estimates.read_text(encoding="utf-8"))
    results = results_mod.assemble(
        config=cfg, estimates=estimates, manifest=Manifest.read(paths.manifest)
    )
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
    *,
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
) -> None:
    """Write synthetic sources in the exact upstream formats, for offline runs and CI."""
    paths = ProjectPaths.discover()
    fixture_mod.write_fixture(config=_config(paths, config, override), mirror=target, seed=seed)
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
        typer.echo(f"stale: {', '.join(stale)}; run `uv run vrp readme-render`", err=True)
        raise typer.Exit(code=1)
    typer.echo("README.md and paper/paper.qmd match results.json")


@app.command("check-provenance")
def check_provenance(
    *,
    rehash: Annotated[
        bool, typer.Option(help="Also hash every raw file on disk again (needs the downloads).")
    ] = False,
) -> None:
    """Fail unless results.json was built from the official sources with the committed config."""
    paths = ProjectPaths.discover()
    cfg = config_mod.load(path=paths.config)
    manifest = Manifest.read(paths.manifest)
    problems = provenance.verify(
        results=results_mod.read(paths.results),
        manifest=manifest,
        official_hosts=cfg.sources.official_hosts,
        expected_fingerprint=cfg.fingerprint(),
    )
    if rehash:
        problems += provenance.rehash(
            manifest=manifest, data_dir=paths.data, workers=cfg.download.workers, say=_say
        )
    if problems:
        for problem in problems[:20]:
            typer.echo(f"provenance: {problem}", err=True)
        if len(problems) > 20:
            typer.echo(f"provenance: and {len(problems) - 20} more", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        "results.json comes from the official sources"
        + (" and every raw file matches its hash" if rehash else "")
    )


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
