"""Fetch new and revised arXiv papers into the local Parquet store.

    python -m rag_papers.ingest --days 30
    python -m rag_papers.ingest --backfill 140

--backfill reaches further back in windows of a week, newest first, and saves after each window.
It only fetches what is older than the store's oldest update, so a run stopped by arXiv's rate
limit continues where it stopped. This assumes the store covers every update from its oldest one
to now, which holds because the daily job fetches the last 3 days every day.
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rag_papers import store
from rag_papers.arxiv_client import ArxivClient

CATEGORY = "cs.CL"
BACKFILL_DAYS = 30
DEFAULT_PATH = Path("data/papers.parquet")
WINDOW = timedelta(days=7)


def run(path: Path, days: int, client: ArxivClient | None = None, now: datetime | None = None) -> int:
    """Upsert papers updated in the last `days` days. Returns the store size."""
    now = now or datetime.now(timezone.utc)
    client = client or ArxivClient()
    records = list(client.fetch_updated_since(CATEGORY, now - timedelta(days=days)))
    df = store.upsert(store.load(path), records)
    store.save(df, path)
    logging.info("fetched %d records, store has %d papers", len(records), len(df))
    return len(df)


def backfill(path: Path, days: int, client: ArxivClient | None = None, now: datetime | None = None) -> int:
    """Fill the store back to `days` ago, one week-long window at a time. Returns the store size."""
    now = now or datetime.now(timezone.utc)
    client = client or ArxivClient()
    since = now - timedelta(days=days)
    df = store.load(path)
    until = df["updated"].min().to_pydatetime() if len(df) else now
    if until <= since:
        logging.info("store already reaches %s", since.date())
        return len(df)
    while until > since:
        start = max(since, until - WINDOW)
        records = list(client.fetch_updated_since(CATEGORY, start, until=until))
        df = store.upsert(df, records)
        store.save(df, path)
        logging.info("%s to %s: %d records, store has %d papers", start.date(), until.date(), len(records), len(df))
        until = start
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=BACKFILL_DAYS)
    parser.add_argument("--backfill", type=int, default=0, metavar="DAYS")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.backfill:
        backfill(args.path, args.backfill)
    else:
        run(args.path, args.days)


if __name__ == "__main__":
    main()
