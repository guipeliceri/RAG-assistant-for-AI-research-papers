"""Client for the arXiv export API."""

import logging
import re
import time
from collections.abc import Iterator
from datetime import datetime, timezone

import feedparser
import requests

from rag_papers.dates import listing_date
from rag_papers.ids import parse_arxiv_id

API_URL = "https://export.arxiv.org/api/query"
# arXiv asks for at most one request every 3 seconds.
REQUEST_DELAY = 3.0
PAGE_SIZE = 200
MAX_RETRIES = 6
# A 429 or 503 means arXiv wants us to slow down for a while, not for a few seconds.
RATE_LIMIT_WAIT = 30.0
MAX_WAIT = 300.0

log = logging.getLogger(__name__)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def parse_entry(entry) -> dict:
    """Turn one Atom entry into a paper record (see store.COLUMNS)."""
    arxiv_id, version = parse_arxiv_id(entry.id)
    published = _timestamp(entry.published)
    return {
        "arxiv_id": arxiv_id,
        "version": version,
        "versions": [version],
        "title": _clean(entry.title),
        "abstract": _clean(entry.summary),
        "authors": [a.name for a in entry.get("authors", [])],
        "categories": [t.term for t in entry.get("tags", [])],
        "primary_category": entry.get("arxiv_primary_category", {}).get("term", ""),
        "published": published,
        "updated": _timestamp(entry.updated),
        "listing_date": listing_date(published),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


class ArxivClient:
    def __init__(self, session: requests.Session | None = None, delay: float = REQUEST_DELAY):
        self.session = session or requests.Session()
        self.delay = delay
        self._last_request = 0.0

    def _get(self, params: dict) -> feedparser.FeedParserDict:
        for attempt in range(MAX_RETRIES):
            wait = self.delay - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                response = self.session.get(API_URL, params=params, timeout=60)
                response.raise_for_status()
                return feedparser.parse(response.text)
            except requests.RequestException as exc:
                log.warning("arXiv request failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(self._backoff(exc, attempt))
        raise RuntimeError("arXiv API unavailable after retries")

    def _backoff(self, exc: requests.RequestException, attempt: int) -> float:
        response = getattr(exc, "response", None)
        if response is not None and response.status_code in (429, 503):
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                return min(float(retry_after), MAX_WAIT)
            return min(RATE_LIMIT_WAIT * 2**attempt, MAX_WAIT)
        return self.delay * 2**attempt

    def _page(self, category: str, start: int, until: datetime | None = None, since: datetime | None = None):
        query = f"cat:{category}"
        if until is not None and since is not None:
            query += f" AND lastUpdatedDate:[{since:%Y%m%d%H%M} TO {until:%Y%m%d%H%M}]"
        return self._get(
            {
                "search_query": query,
                "sortBy": "lastUpdatedDate",
                "sortOrder": "descending",
                "start": start,
                "max_results": PAGE_SIZE,
            }
        )

    def fetch_updated_since(self, category: str, since: datetime, until: datetime | None = None) -> Iterator[dict]:
        """Yield every paper in `category` created or revised at or after `since` (and before `until`).

        Results come newest-update first, so paging stops at the first entry
        older than `since`. With `until`, the query itself is limited to that
        range, which keeps paging shallow: arXiv rate-limits deep pages.
        """
        start = 0
        while True:
            feed = self._page(category, start, until, since if until else None)
            if not feed.entries and start < int(feed.feed.get("opensearch_totalresults", 0)):
                # The API sometimes returns an empty page mid-listing; retry once.
                feed = self._page(category, start, until, since if until else None)
            if not feed.entries:
                return
            for entry in feed.entries:
                record = parse_entry(entry)
                if record["updated"] < since:
                    return
                if until is not None and record["updated"] >= until:
                    continue
                yield record
            start += len(feed.entries)
