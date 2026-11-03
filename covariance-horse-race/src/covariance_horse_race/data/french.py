"""Parser for the Kenneth French daily CSV layout.

A file holds free-text preamble lines, then one or more blocks. Each block is an
optional title line ("Average Value Weighted Returns -- Daily"), a header line
whose first field is empty, and rows keyed by YYYYMMDD dates. Blocks are
separated by blank or text lines; a copyright line may close the file.
The parser finds blocks structurally instead of by line numbers, so preamble
changes upstream do not shift the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from covariance_horse_race.data.source import SourceError

_DATE = re.compile(r"^\s*(\d{8})\s*$")
_SECTION_PATTERNS = {
    "value_weighted": re.compile(r"value\s+weighted", re.IGNORECASE),
    "equal_weighted": re.compile(r"equal\s+weighted", re.IGNORECASE),
}


@dataclass(frozen=True)
class Block:
    title: str
    columns: list[str]
    dates: list[str]
    rows: list[list[str]]


def decode(payload: bytes) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1")


def _fields(line: str) -> list[str]:
    fields = [field.strip() for field in line.split(",")]
    while len(fields) > 1 and fields[-1] == "":
        fields.pop()
    return fields


def split_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    last_text = ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        fields = _fields(lines[i])
        is_header = len(fields) > 1 and fields[0] == "" and all(fields[1:])
        if not is_header:
            stripped = lines[i].strip()
            if stripped:
                last_text = stripped
            i += 1
            continue
        columns = fields[1:]
        dates: list[str] = []
        rows: list[list[str]] = []
        i += 1
        while i < len(lines):
            row = _fields(lines[i])
            if not _DATE.match(row[0]):
                break
            if len(row) - 1 != len(columns):
                raise SourceError(
                    f"row {row[0]} has {len(row) - 1} values for {len(columns)} columns"
                )
            dates.append(row[0])
            rows.append(row[1:])
            i += 1
        if dates:
            blocks.append(Block(title=last_text, columns=columns, dates=dates, rows=rows))
        last_text = ""
    return blocks


def select_block(blocks: list[Block], section: str | None) -> Block:
    """Pick the requested weighting section, or the only block for factor files."""
    if not blocks:
        raise SourceError("no data blocks found")
    if section is None:
        if len(blocks) != 1:
            titles = [b.title for b in blocks]
            raise SourceError(f"expected one block, found {len(blocks)}: {titles}")
        return blocks[0]
    pattern = _SECTION_PATTERNS[section]
    matches = [b for b in blocks if pattern.search(b.title) and "daily" in b.title.lower()]
    if len(matches) != 1:
        titles = [b.title for b in blocks]
        raise SourceError(f"section {section!r} matched {len(matches)} blocks among {titles}")
    return matches[0]


def to_frame(block: Block, sentinels: list[float], percent_scale: float) -> pd.DataFrame:
    """Convert a block to decimal returns indexed by date, sentinels as NaN."""
    try:
        values = np.array(block.rows, dtype=float)
    except ValueError as exc:
        raise SourceError(f"non-numeric value in block {block.title!r}") from exc
    for sentinel in sentinels:
        values[np.isclose(values, sentinel)] = np.nan
    index = pd.DatetimeIndex(pd.to_datetime(block.dates, format="%Y%m%d"), name="date")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise SourceError(f"dates in block {block.title!r} are not strictly increasing")
    if len(set(block.columns)) != len(block.columns):
        raise SourceError(f"duplicate column names in block {block.title!r}")
    frame = pd.DataFrame(values / percent_scale, index=index, columns=block.columns)
    return frame


def parse(
    payload: bytes, section: str | None, sentinels: list[float], scale: float
) -> pd.DataFrame:
    return to_frame(select_block(split_blocks(decode(payload)), section), sentinels, scale)
