"""Detect a newer OSAP release branch on Hugging Face and pin it in the config."""

from __future__ import annotations

import re
from pathlib import Path

import requests

_PATTERNS = (re.compile(r"^Data-Release-(\d{4})\.(\d{2})$"), re.compile(r"^(\d{4})\.(\d{2})$"))


def release_key(name: str) -> tuple[int, int] | None:
    for pattern in _PATTERNS:
        match = pattern.match(name)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def newest_from(names: list[str]) -> str:
    dated = [(key, name) for name in names if (key := release_key(name)) is not None]
    if not dated:
        raise ValueError("no release branches found")
    return max(dated)[1]


def newest_release(api: str, *, timeout: float = 30.0) -> str:
    response = requests.get(f"{api}/refs", timeout=timeout)
    response.raise_for_status()
    branches = [b["name"] for b in response.json().get("branches", [])]
    return newest_from(branches)


def pin(config_path: Path, *, old: str, new: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    needle = f"osap_revision: {old}"
    if needle not in text:
        raise ValueError(f"{config_path} does not contain {needle!r}")
    config_path.write_text(text.replace(needle, f"osap_revision: {new}", 1), encoding="utf-8")
