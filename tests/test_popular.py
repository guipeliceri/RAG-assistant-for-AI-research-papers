from datetime import date, datetime, timezone

import pandas as pd
import pytest
import requests

from rag_papers import popular

NOW = datetime(2026, 10, 26, 2, 0, tzinfo=timezone.utc)


def item(arxiv_id, upvotes, day, comments=0, **paper):
    return {
        "paper": {
            "id": arxiv_id,
            "title": f"Paper\n  {arxiv_id}",
            "summary": f"Abstract {arxiv_id}",
            "authors": [{"_id": "x", "name": "Ada"}, {"_id": "y", "name": "Alan"}],
            "upvotes": upvotes,
            "submittedOnDailyAt": f"{day}T00:00:00.000Z",
            **paper,
        },
        "numComments": comments,
    }


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.payload


class FakeSession:
    """Serves one list per date; dates it does not know return an empty list."""

    def __init__(self, by_day, fail_first=0):
        self.by_day = by_day
        self.fail_first = fail_first
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params["date"])
        if self.fail_first:
            self.fail_first -= 1
            return FakeResponse(None, 503)
        return FakeResponse(self.by_day.get(params["date"], []))


def test_parse_reads_votes_and_links():
    row = popular.parse(item("2610.1", 42, "2026-10-22", comments=3, ai_summary="Short."), NOW)
    assert row["title"] == "Paper 2610.1"
    assert row["upvotes"] == 42 and row["comments"] == 3
    assert row["daily_date"] == date(2026, 10, 22)
    assert row["authors"] == ["Ada", "Alan"]
    assert row["summary"] == "Short."  # HF's short summary is preferred over the abstract
    assert row["hf_url"] == "https://huggingface.co/papers/2610.1"


def test_update_keeps_latest_vote_count():
    first = popular.update(pd.DataFrame(columns=popular.COLUMNS), [popular.parse(item("a", 5, "2026-10-22"), NOW)])
    later = popular.update(first, [popular.parse(item("a", 50, "2026-10-22"), NOW), popular.parse(item("b", 1, "2026-10-23"), NOW)])
    assert len(later) == 2
    assert later.set_index("arxiv_id").loc["a", "upvotes"] == 50


def test_top_filters_by_period_and_sorts_by_votes():
    rows = [
        popular.parse(item("old", 999, "2026-10-01"), NOW),
        popular.parse(item("low", 3, "2026-10-20"), NOW),
        popular.parse(item("tie-more-comments", 10, "2026-10-21", comments=5), NOW),
        popular.parse(item("tie", 10, "2026-10-22", comments=1), NOW),
    ]
    df = popular.update(pd.DataFrame(columns=popular.COLUMNS), rows)
    ranked = popular.top(df, date(2026, 10, 15), date(2026, 10, 25), n=10)
    assert list(ranked["arxiv_id"]) == ["tie-more-comments", "tie", "low"]
    assert list(popular.top(df, None, None, n=1)["arxiv_id"]) == ["old"]


def test_run_refetches_recent_days_and_saves(tmp_path):
    session = FakeSession({"2026-10-25": [item("a", 7, "2026-10-25")], "2026-10-26": [item("b", 2, "2026-10-26")]})
    path = tmp_path / "popular.parquet"
    assert popular.run(path, days=2, today=date(2026, 10, 26), session=session, delay=0) == 2
    assert session.calls == ["2026-10-24", "2026-10-25", "2026-10-26"]
    saved = pd.read_parquet(path)
    assert set(saved["arxiv_id"]) == {"a", "b"}


def test_fetch_day_retries_then_raises(monkeypatch):
    monkeypatch.setattr(popular.time, "sleep", lambda s: None)
    ok = FakeSession({"2026-10-22": [item("a", 1, "2026-10-22")]}, fail_first=1)
    assert [r["arxiv_id"] for r in popular.fetch_day(date(2026, 10, 22), ok, NOW)] == ["a"]
    with pytest.raises(requests.HTTPError):
        popular.fetch_day(date(2026, 10, 22), FakeSession({}, fail_first=popular.RETRIES), NOW)
