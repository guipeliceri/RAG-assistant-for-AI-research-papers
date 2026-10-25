"""Fetch new and revised arXiv papers into the local Parquet store.

    python -m rag_papers.ingest --days 30
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


def run(path: Path, days: int, client: ArxivClient | None = None, now: datetime | None = None) -> int:
    """Upsert papers updated in the last `days` days. Returns the store size."""
    now = now or datetime.now(timezone.utc)
    client = client or ArxivClient()
    records = list(client.fetch_updated_since(CATEGORY, now - timedelta(days=days)))
    df = store.upsert(store.load(path), records)
    store.save(df, path)
    logging.info("fetched %d records, store has %d papers", len(records), len(df))
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=BACKFILL_DAYS)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.path, args.days)


if __name__ == "__main__":
    main()
