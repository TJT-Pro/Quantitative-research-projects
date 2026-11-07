"""Render README.md and paper/paper.qmd from results.json through Jinja templates.

The README uses standard Jinja delimiters. The paper uses << >> and <% %> so Quarto's
own braces ({#fig-x}, {{< pagebreak >}}) pass through untouched.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jinja2

from cross_asset_vrp.paths import ProjectPaths


def _num(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def _pct(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"


def _signed(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def _ordinal(value: float) -> str:
    n = round(value)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _inline(text: str) -> str:
    """A label as it reads mid-sentence: 'Gold' becomes 'gold', 'US equities' stays."""
    first = text.split(" ", 1)[0]
    return text if first.isupper() else text[:1].lower() + text[1:]


def _join_words(items: list[str]) -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _environment(*, paper: bool) -> jinja2.Environment:
    options: dict[str, Any] = {
        "undefined": jinja2.StrictUndefined,
        "keep_trailing_newline": True,
        "trim_blocks": True,
        "lstrip_blocks": True,
        "autoescape": False,  # Markdown and Quarto output, not HTML
    }
    if paper:
        options.update(
            variable_start_string="<<",
            variable_end_string=">>",
            block_start_string="<%",
            block_end_string="%>",
            comment_start_string="<#",
            comment_end_string="#>",
        )
    env = jinja2.Environment(**options)  # noqa: S701 - see autoescape above
    env.filters.update(
        num=_num, pct=_pct, signed=_signed, join_words=_join_words, ordinal=_ordinal, inline=_inline
    )
    return env


def render(template: Path, results: dict[str, Any], *, paper: bool) -> str:
    env = _environment(paper=paper)
    return env.from_string(template.read_text(encoding="utf-8")).render(r=results)


def _pairs(paths: ProjectPaths) -> list[tuple[Path, Path, bool]]:
    return [
        (paths.readme_template, paths.readme, False),
        (paths.paper_template, paths.paper, True),
    ]


def render_all(paths: ProjectPaths, results: dict[str, Any]) -> None:
    for template, target, paper in _pairs(paths):
        target.write_text(render(template, results, paper=paper), encoding="utf-8")


def stale(paths: ProjectPaths, results: dict[str, Any]) -> list[str]:
    out = []
    for template, target, paper in _pairs(paths):
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        if render(template, results, paper=paper) != current:
            out.append(str(target.relative_to(paths.root)))
    return out
