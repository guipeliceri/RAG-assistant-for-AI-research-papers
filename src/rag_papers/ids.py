"""arXiv identifier handling."""

import re

# New style: 2410.12345v2. Old style: cs/0101001v1, math.GT/0309136v3.
_ID_RE = re.compile(
    r"(?P<base>\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v(?P<version>\d+))?$"
)


def parse_arxiv_id(raw: str) -> tuple[str, int]:
    """Split an arXiv ID or abs URL into (ID without version, version number).

    >>> parse_arxiv_id("http://arxiv.org/abs/2410.12345v2")
    ('2410.12345', 2)
    """
    tail = raw.strip().rstrip("/").split("/abs/")[-1]
    match = _ID_RE.search(tail)
    if not match:
        raise ValueError(f"not an arXiv ID: {raw!r}")
    return match.group("base"), int(match.group("version") or 1)
