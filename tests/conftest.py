from pathlib import Path

import pytest

from rag_papers.arxiv_client import ArxivClient

FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


class FakeSession:
    """Serves the given Atom pages in order, then empty feeds."""

    def __init__(self, pages: list[str]):
        self.pages = list(pages)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        text = self.pages.pop(0) if self.pages else EMPTY_FEED
        return FakeResponse(text)


EMPTY_FEED = (
    '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" '
    'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
    "<opensearch:totalResults>0</opensearch:totalResults></feed>"
)


@pytest.fixture
def feed_page() -> str:
    return (FIXTURES / "feed_page.xml").read_text()


@pytest.fixture
def make_client():
    """Build an ArxivClient that serves the given Atom pages, with no delay."""

    def make(*pages: str) -> ArxivClient:
        return ArxivClient(session=FakeSession(list(pages)), delay=0)

    return make
