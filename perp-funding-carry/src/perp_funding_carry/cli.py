"""pfc: the perp-funding-carry command line. Every stage is one command; `results` runs them all."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from perp_funding_carry import config as config_mod
from perp_funding_carry.data import books as books_mod
from perp_funding_carry.data import build as build_mod
from perp_funding_carry.data import fetch as fetch_mod
from perp_funding_carry.data import fixture as fixture_mod
from perp_funding_carry.data.manifest import Manifest
from perp_funding_carry.evaluation import pipeline
from perp_funding_carry.paths import ProjectPaths
from perp_funding_carry.plotting import figures
from perp_funding_carry.report import provenance, render
from perp_funding_carry.report import results as results_mod

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
    """Catalogue the bucket, download what changed, collect Bybit, OKX and Binance REST data."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    s = fetch_mod.fetch_all(config=cfg, paths=paths, mirror=mirror)
    typer.echo(
        f"{s.contracts} contracts, {s.files:,} files to {s.last_month} "
        f"({s.requested:,} requested, {s.changed:,} new or changed); "
        f"{s.okx_rows_added} new OKX prints in the local store"
    )


@app.command()
def snapshot(
    *, config: ConfigOpt = None, override: OverrideOpt = None, mirror: MirrorOpt = None
) -> None:
    """Record the best bid, ask and 24-hour volume of every spot and USD-M pair (two requests
    per market, a few seconds). The spread model is fitted to every snapshot stored so far."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    manifest = Manifest.read(paths.manifest)
    client = fetch_mod.client_for(config=cfg, paths=paths, manifest=manifest, mirror=mirror)
    taken = books_mod.snapshot(cfg, client, data_dir=paths.data)
    manifest.write(paths.manifest)
    typer.echo(
        f"{taken.rows_added:,} book rows added; {taken.snapshots} snapshot"
        f"{'' if taken.snapshots == 1 else 's'} stored"
    )


@app.command()
def build(*, config: ConfigOpt = None, override: OverrideOpt = None) -> None:
    """Parse data/raw into Parquet tables in data/processed."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    tables = build_mod.build_tables(config=cfg, paths=paths)
    build_mod.write_tables(tables, paths.processed)
    delisted = int(tables.contracts["delisted"].sum())
    typer.echo(
        f"{len(tables.contracts)} contracts ({delisted} delisted), "
        f"{len(tables.funding):,} funding prints"
    )


@app.command()
def estimate(*, config: ConfigOpt = None, override: OverrideOpt = None) -> None:
    """Run every analysis; writes data/runs/estimates.json and reports/tables/*.csv."""
    paths = ProjectPaths.discover()
    cfg = _config(paths, config, override)
    tables = build_mod.read_tables(paths.processed)
    snapshots = books_mod.read_store(paths.data / books_mod.STORE)
    estimates = pipeline.run(
        config=cfg, tables=tables, snapshots=snapshots, tables_dir=paths.tables, log=True
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
        typer.echo(f"stale: {', '.join(stale)}; run `uv run pfc readme-render`", err=True)
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
            manifest=manifest,
            data_dir=paths.data,
            workers=cfg.download.workers,
            say=typer.echo,
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
