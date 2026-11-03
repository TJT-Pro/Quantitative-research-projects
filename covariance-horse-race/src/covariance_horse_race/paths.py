"""Filesystem layout of one project, resolved from its root directory."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME = "covariance-horse-race"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> ProjectPaths:
        """Walk up from `start` to the directory whose pyproject names this project."""
        here = (start or Path.cwd()).resolve()
        for candidate in (here, *here.parents):
            pyproject = candidate / "pyproject.toml"
            if pyproject.is_file():
                with pyproject.open("rb") as handle:
                    name = tomllib.load(handle).get("project", {}).get("name")
                if name == PROJECT_NAME:
                    return cls(candidate)
        raise FileNotFoundError(f"no pyproject.toml for {PROJECT_NAME!r} above {here}")

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def manifest(self) -> Path:
        return self.root / "data" / "manifest.json"

    @property
    def runs(self) -> Path:
        return self.root / "data" / "runs"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "reports" / "tables"

    @property
    def results(self) -> Path:
        return self.root / "reports" / "results.json"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    @property
    def readme_template(self) -> Path:
        return self.root / "README.template.md"

    @property
    def paper_template(self) -> Path:
        return self.root / "paper" / "paper.template.qmd"

    @property
    def paper(self) -> Path:
        return self.root / "paper" / "paper.qmd"

    def ensure(self) -> None:
        for directory in (self.raw, self.processed, self.runs, self.figures, self.tables):
            directory.mkdir(parents=True, exist_ok=True)
