"""Most-voted papers: community upvotes from Hugging Face Daily Papers.

    python -m rag_papers.popular [--days 30] [--path data/popular.parquet]

arXiv publishes no view or rating counts. Daily Papers is the public signal that exists: each day
the community submits and upvotes a few dozen AI papers (any area, not only cs.CL). Votes keep
coming in for days, so every run re-fetches the recent days and overwrites the counts.
"""

import argparse
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from rag_papers import store

API_URL = "https://huggingface.co/api/daily_papers"
DEFAULT_PATH = Path("data/popular.parquet")
REQUEST_DELAY = 1.0
RETRIES = 3

COLUMNS = [
    "arxiv_id",
    "title",
    "summary",
    "authors",
    # The day the paper was featured on Daily Papers; this is what the date filter uses.
    "daily_date",
    "upvotes",
    "comments",
    "github_repo",
    "organization",
    "url",
    "hf_url",
    "fetched_at",
]


def parse(item: dict, fetched_at: datetime, day: date | None = None) -> dict:
    paper = item.get("paper", {})
    arxiv_id = paper["id"]
    featured = paper.get("submittedOnDailyAt") or (day.isoformat() if day else item["publishedAt"])
    return {
        "arxiv_id": arxiv_id,
        "title": " ".join((paper.get("title") or item.get("title") or "").split()),
        "summary": (paper.get("ai_summary") or paper.get("summary") or "").strip(),
        "authors": [a["name"] for a in paper.get("authors", []) if a.get("name")],
        "daily_date": date.fromisoformat(featured[:10]),
        "upvotes": int(paper.get("upvotes") or 0),
        "comments": int(item.get("numComments") or 0),
        "github_repo": paper.get("githubRepo"),
        "organization": (paper.get("organization") or item.get("organization") or {}).get("fullname"),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "hf_url": f"https://huggingface.co/papers/{arxiv_id}",
        "fetched_at": fetched_at,
    }


def fetch_day(day: date, session=None, now: datetime | None = None) -> list[dict]:
    session = session or requests.Session()
    now = now or datetime.now(timezone.utc)
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(API_URL, params={"date": day.isoformat()}, timeout=30)
            response.raise_for_status()
            return [parse(item, now, day) for item in response.json() if item.get("paper", {}).get("id")]
        except (requests.RequestException, ValueError) as exc:
            if attempt == RETRIES:
                raise
            logging.warning("daily papers %s failed (%s), retrying", day, exc)
            time.sleep(2**attempt)
    return []


def update(df: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """One row per paper; the latest fetch wins, so vote counts only move forward in time."""
    if not records:
        return df
    new = pd.DataFrame(records, columns=COLUMNS)
    merged = pd.concat([df, new], ignore_index=True) if len(df) else new
    return merged.drop_duplicates("arxiv_id", keep="last").reset_index(drop=True)


def top(df: pd.DataFrame, start: date | None, end: date | None, n: int = 10) -> pd.DataFrame:
    """Most-upvoted papers featured between `start` and `end` (inclusive), ties broken by comments."""
    if df.empty:
        return df
    days = pd.to_datetime(df["daily_date"]).dt.date
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= days >= start
    if end:
        mask &= days <= end
    return df[mask].sort_values(["upvotes", "comments"], ascending=False).head(n)


def run(path: Path, days: int, today: date | None = None, session=None, delay: float = REQUEST_DELAY) -> int:
    today = today or datetime.now(timezone.utc).date()
    df = store.load(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    total = 0
    for offset in range(days, -1, -1):
        day = today - timedelta(days=offset)
        records = fetch_day(day, session)
        df = update(df, records)
        total += len(records)
        if delay:
            time.sleep(delay)
    store.save(df, path)
    logging.info("popular: %d entries fetched over %d days, %d papers stored", total, days + 1, len(df))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30, help="how many past days to (re)fetch")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.path, args.days)


if __name__ == "__main__":
    main()
