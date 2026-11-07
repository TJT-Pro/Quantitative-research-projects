"""The Binance Vision bucket listing: which symbols and months exist, delisted ones included.

The listing is S3 ListObjects (version 1) XML, paged with `marker`. Each object carries
its size and ETag; for these single-part objects the ETag is the MD5 of the file, which
the downloader uses to skip unchanged files and to verify new ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from defusedxml import ElementTree

from perp_funding_carry.data.download import Client


@dataclass(frozen=True)
class ListedObject:
    key: str
    size: int
    md5: str | None  # None for multipart uploads, whose ETag is not an MD5


@dataclass(frozen=True)
class Listing:
    prefixes: list[str]
    objects: list[ListedObject]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def page_url(base: str, *, prefix: str, marker: str) -> str:
    url = f"{base}?delimiter=/&prefix={quote(prefix)}"
    return url + (f"&marker={quote(marker)}" if marker else "")


def parse_page(text: str) -> tuple[Listing, bool, str]:
    root = ElementTree.fromstring(text)
    prefixes: list[str] = []
    objects: list[ListedObject] = []
    truncated = False
    next_marker = ""
    for child in root:
        name = _local(child.tag)
        if name == "IsTruncated":
            truncated = (child.text or "").strip().lower() == "true"
        elif name == "NextMarker":
            next_marker = child.text or ""
        elif name == "CommonPrefixes":
            for item in child:
                if _local(item.tag) == "Prefix" and item.text:
                    prefixes.append(item.text)
        elif name == "Contents":
            fields = {_local(item.tag): item.text or "" for item in child}
            etag = fields.get("ETag", "").strip('"')
            objects.append(
                ListedObject(
                    key=fields["Key"],
                    size=int(fields.get("Size", "0")),
                    md5=etag if len(etag) == 32 and "-" not in etag else None,
                )
            )
    if truncated and not next_marker:
        last = [*prefixes, *(o.key for o in objects)]
        next_marker = max(last) if last else ""
    return Listing(prefixes=prefixes, objects=objects), truncated, next_marker


def list_prefix(client: Client, base: str, prefix: str) -> Listing:
    prefixes: list[str] = []
    objects: list[ListedObject] = []
    marker = ""
    while True:
        listing, truncated, marker = parse_page(
            client.get_text(page_url(base, prefix=prefix, marker=marker))
        )
        prefixes += listing.prefixes
        objects += listing.objects
        if not truncated:
            return Listing(prefixes=prefixes, objects=objects)
