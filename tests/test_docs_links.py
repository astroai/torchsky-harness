"""Keep non-archive documentation links resolvable.

The archive keeps historical snapshots with their original relative links;
everything else must point at a file that exists in this repository or at an
absolute URL.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#", "data:")


def _broken_links() -> list[str]:
    broken: list[str] = []
    for page in sorted(DOCS.rglob("*.md")):
        if "archive" in page.parts:
            continue
        for match in LINK.finditer(page.read_text(encoding="utf-8")):
            link = match.group(1)
            if link.startswith(SKIP_PREFIXES):
                continue
            target = link.split("#", 1)[0].split("?", 1)[0]
            if target and not (page.parent / target).resolve().exists():
                broken.append(f"{page.relative_to(DOCS.parent)} -> {link}")
    return broken


def test_non_archive_doc_links_resolve() -> None:
    broken = _broken_links()
    assert not broken, "broken documentation links:\n" + "\n".join(broken)
