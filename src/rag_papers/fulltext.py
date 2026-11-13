"""Introduction, conclusion and affiliations of a paper's first version, from arXiv's HTML.

Always v1 (`https://arxiv.org/html/<id>v1`): a later version can carry information that did not
exist when the paper was announced. Not every paper has an HTML version; then the forecast
falls back to the abstract and says so.
"""

import logging
import re
import time
from html.parser import HTMLParser

import requests

from rag_papers.hype import FullText

HTML_URL = "https://arxiv.org/html/{id}v1"
MAX_SECTION_CHARS = 2500
MAX_AFFILIATION_CHARS = 400
REQUEST_DELAY = 3.0


class _Sections(HTMLParser):
    """Collects text per top-level section heading, plus author affiliation text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections: dict[str, list[str]] = {}
        self.affiliations: list[str] = []
        self._heading: list[str] | None = None
        self._current: str | None = None
        self._affil_depth = 0
        self._depth = 0
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        self._depth += 1
        cls = dict(attrs).get("class") or ""
        if tag in ("script", "style", "math", "figure", "table"):
            self._skip += 1
        if tag == "h2":
            self._heading = []
        if "ltx_role_affiliation" in cls or "ltx_affiliation" in cls:
            self._affil_depth = self._depth

    def handle_endtag(self, tag):
        if tag in ("script", "style", "math", "figure", "table") and self._skip:
            self._skip -= 1
        if tag == "h2" and self._heading is not None:
            title = re.sub(r"^(?:[\dIVXLC]+\.?\s+)+", "", " ".join(self._heading)).strip().lower()
            self._current = title
            self.sections.setdefault(title, [])
            self._heading = None
        if self._affil_depth and self._depth <= self._affil_depth:
            self._affil_depth = 0
            self.affiliations.append("|")
        self._depth -= 1

    def handle_data(self, data):
        if self._skip:
            return
        if self._heading is not None:
            self._heading.append(data.strip())
        elif self._affil_depth:
            self.affiliations.append(data.strip())
        elif self._current is not None:
            self.sections[self._current].append(data)


def _clean(parts: list[str], limit: int) -> str:
    return re.sub(r"\s+", " ", " ".join(parts)).strip()[:limit]


def parse(html: str) -> FullText:
    p = _Sections()
    p.feed(html)
    intro = next((v for k, v in p.sections.items() if k.startswith("introduction")), [])
    concl = next((v for k, v in p.sections.items() if k.startswith(("conclusion", "discussion"))), [])
    affiliations = "; ".join(x for x in " ".join(p.affiliations).split("|") if x.strip())
    return FullText(
        introduction=_clean(intro, MAX_SECTION_CHARS),
        conclusion=_clean(concl, MAX_SECTION_CHARS),
        affiliations=_clean([affiliations], MAX_AFFILIATION_CHARS),
        source="arXiv HTML v1" if intro or concl else "abstract only",
    )


def fetch(arxiv_id: str, session: requests.Session | None = None, delay: float = REQUEST_DELAY) -> FullText:
    session = session or requests.Session()
    try:
        response = session.get(HTML_URL.format(id=arxiv_id), timeout=30)
    except requests.RequestException as exc:
        logging.warning("no HTML for %s: %s", arxiv_id, type(exc).__name__)
        return FullText()
    finally:
        if delay:
            time.sleep(delay)
    if not response.ok:
        return FullText()
    return parse(response.text)
