"""Render README.md from README.template.md and results.json.

The template holds prose; every number, list of models and figure path comes
from results.json. StrictUndefined makes a missing value an error instead of an
empty string, and `check` lets CI fail when the committed README is stale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined


def _pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def _num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


_LOSS_NAMES = {
    "gmv_variance": "GMV variance",
    "gmv_long_only_variance": "long-only GMV variance",
    "qlike": "QLIKE",
    "frobenius": "Frobenius",
}


def _loss_name(key: str) -> str:
    return _LOSS_NAMES.get(key, key.replace("_", " "))


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def environment(*, quarto: bool = False) -> Environment:
    """Jinja environment for Markdown output.

    Quarto uses `{#id}` for cross-references, which is Jinja's comment opener, so
    Quarto templates use `<#-- ... --#>` for comments instead.
    """
    comment_start, comment_end = ("<#--", "--#>") if quarto else ("{#", "#}")
    env = Environment(
        undefined=StrictUndefined,
        # Renders Markdown from our own results.json, never HTML or untrusted input.
        autoescape=False,  # noqa: S701
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        comment_start_string=comment_start,
        comment_end_string=comment_end,
    )
    env.filters["pct"] = _pct
    env.filters["num"] = _num
    env.filters["join_words"] = _join
    env.filters["loss_name"] = _loss_name
    return env


def render(template_path: Path, results: dict[str, Any]) -> str:
    quarto = template_path.suffix == ".qmd"
    template = environment(quarto=quarto).from_string(template_path.read_text(encoding="utf-8"))
    return template.render(r=results)


def write(template_path: Path, results: dict[str, Any], output: Path) -> None:
    output.write_text(render(template_path, results), encoding="utf-8")


def is_current(template_path: Path, results: dict[str, Any], output: Path) -> bool:
    if not output.is_file():
        return False
    return output.read_text(encoding="utf-8") == render(template_path, results)
