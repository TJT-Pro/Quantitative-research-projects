"""Every file location the pipeline reads or writes, resolved from the project root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> ProjectPaths:
        """Walk up from `start` (default: the working directory) to the project folder."""
        here = (start or Path.cwd()).resolve()
        for candidate in (here, *here.parents):
            pyproject = candidate / "pyproject.toml"
            if pyproject.is_file() and 'name = "factor-zoo-trial"' in pyproject.read_text(
                encoding="utf-8"
            ):
                return cls(root=candidate)
        raise FileNotFoundError("run fzt from inside the factor-zoo-trial folder")

    @property
    def config(self) -> Path:
        return self.root / "configs" / "research.yaml"

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def runs(self) -> Path:
        return self.root / "data" / "runs"

    @property
    def manifest(self) -> Path:
        return self.root / "data" / "manifest.json"

    @property
    def estimates(self) -> Path:
        return self.runs / "estimates.json"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def tables(self) -> Path:
        return self.reports / "tables"

    @property
    def results(self) -> Path:
        return self.reports / "results.json"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    @property
    def readme_template(self) -> Path:
        return self.root / "README.template.md"

    @property
    def paper(self) -> Path:
        return self.root / "paper" / "paper.qmd"

    @property
    def paper_template(self) -> Path:
        return self.root / "paper" / "paper.template.qmd"
