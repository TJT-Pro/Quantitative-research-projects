"""Command line: `acm <command>`. Works the same in PowerShell, cmd, bash and zsh."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from acm_term_premium import pipeline
from acm_term_premium.config import ResearchConfig, load_config
from acm_term_premium.data.fixture import write_fixture
from acm_term_premium.paths import ProjectPaths
from acm_term_premium.report import render
from acm_term_premium.report import results as results_mod

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="The ACM Treasury term premium, rebuilt and stress-tested.",
)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Base research config.")]
OverrideOpt = Annotated[
    list[Path] | None, typer.Option("--override", "-o", help="Override files, applied in order.")
]
_DEFAULT = Path("configs/research.yaml")


def _load(config: Path, override: list[Path] | None) -> tuple[ResearchConfig, ProjectPaths]:
    paths = ProjectPaths.discover()

    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else paths.root / p

    return load_config(resolve(config), [resolve(o) for o in override or []]), paths


def _render_all(paths: ProjectPaths, results: dict[str, object]) -> None:
    render.write(paths.readme_template, results, paths.readme)
    render.write(paths.paper_template, results, paths.paper)


@app.command()
def fetch(
    config: ConfigOpt = _DEFAULT,
    override: OverrideOpt = None,
    mirror: Annotated[
        Path | None, typer.Option(help="Local folder instead of the network.")
    ] = None,
) -> None:
    """Download every source, verify it and update data/manifest.json."""
    cfg, paths = _load(config, override)
    report = pipeline.fetch(cfg, paths, mirror, typer.echo)
    typer.echo(
        f"fetched {len(report.entries)} files; changed: {', '.join(report.changed) or 'none'}"
    )
    (paths.root / "data" / "changed.txt").write_text(
        "\n".join(report.changed) + "\n", encoding="utf-8"
    )


@app.command()
def build(config: ConfigOpt = _DEFAULT, override: OverrideOpt = None) -> None:
    """Parse the raw files into the monthly and daily curves and policy dates."""
    cfg, paths = _load(config, override)
    pipeline.build(cfg, paths, typer.echo)


@app.command()
def estimate(config: ConfigOpt = _DEFAULT, override: OverrideOpt = None) -> None:
    """Full-sample, bootstrap, real-time, forecasting and policy-day estimates."""
    cfg, paths = _load(config, override)
    pipeline.estimate(cfg, paths, typer.echo)


@app.command()
def report(config: ConfigOpt = _DEFAULT, override: OverrideOpt = None) -> None:
    """Figures, tables, results.json, README.md and the paper source."""
    cfg, paths = _load(config, override)
    results = results_mod.build_results(cfg, paths)
    results_mod.write(results, paths.results)
    _render_all(paths, results)
    typer.echo("wrote reports/results.json, README.md and paper/paper.qmd")


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
    config: ConfigOpt = _DEFAULT,
    override: OverrideOpt = None,
    skip_paper: Annotated[bool, typer.Option(help="Do not build the PDF.")] = False,
) -> None:
    """Everything: download, build, estimate, report and (unless skipped) the paper."""
    fetch(config, override, None)
    build(config, override)
    estimate(config, override)
    report(config, override)
    if not skip_paper:
        paper()


@app.command("readme-render")
def readme_render() -> None:
    """Re-render README.md and paper/paper.qmd from the committed results.json."""
    paths = ProjectPaths.discover()
    _render_all(paths, results_mod.read(paths.results))
    typer.echo("rendered README.md and paper/paper.qmd")


@app.command("check-readme")
def check_readme() -> None:
    """Fail if README.md or the paper source differ from a fresh render."""
    paths = ProjectPaths.discover()
    results = results_mod.read(paths.results)
    stale = [
        out.name
        for tpl, out in ((paths.readme_template, paths.readme), (paths.paper_template, paths.paper))
        if not render.is_current(tpl, results, out)
    ]
    if stale:
        typer.echo(f"stale: {', '.join(stale)}; run `acm readme-render`", err=True)
        raise typer.Exit(code=1)
    typer.echo("README.md and paper/paper.qmd match results.json")


@app.command("check-provenance")
def check_provenance() -> None:
    """Fail unless results.json was built from the official sources, not the fixture."""
    paths = ProjectPaths.discover()
    source = results_mod.read(paths.results).get("provenance")
    if source != "official":
        typer.echo(
            f"results.json provenance is {source!r}; publish only official results", err=True
        )
        raise typer.Exit(code=1)
    typer.echo("results.json comes from the official sources")


@app.command()
def fixture(
    target: Annotated[Path, typer.Argument(help="Folder to write the synthetic sources into.")],
    config: ConfigOpt = _DEFAULT,
    override: OverrideOpt = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
) -> None:
    """Write synthetic files in the exact source formats, for offline runs and CI."""
    cfg, _ = _load(config, override)
    write_fixture(target, cfg, seed)
    typer.echo(f"wrote fixture sources to {target}")


def _run(args: list[str], cwd: Path) -> None:
    typer.echo(f"> {' '.join(args)}")
    completed = subprocess.run(args, cwd=cwd, check=False)  # noqa: S603 - fixed argument lists
    if completed.returncode != 0:
        raise typer.Exit(code=completed.returncode)


@app.command()
def check() -> None:
    """Lint, formatting, strict types, tests and README drift."""
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
        req = Path(tmp) / "requirements.txt"
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
                str(req),
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
                str(req),
            ],
            paths.root,
        )


if __name__ == "__main__":
    app()
