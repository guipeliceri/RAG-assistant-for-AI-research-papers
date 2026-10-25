"""Parquet paper store with upsert keyed on the version-less arXiv ID."""

from pathlib import Path

import pandas as pd

COLUMNS = [
    "arxiv_id",
    "version",
    "versions",
    "title",
    "abstract",
    "authors",
    "categories",
    "primary_category",
    "published",
    "updated",
    "listing_date",
    "url",
]


def load(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=COLUMNS)


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def upsert(df: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Merge `records` into `df`, one row per arXiv ID.

    The newest version wins for the content fields, every version seen is kept
    in `versions`, and `published`/`listing_date` stay those of v1 so a revision
    never makes a paper look new.
    """
    rows = {row["arxiv_id"]: row for row in df.to_dict("records")}
    for record in records:
        current = rows.get(record["arxiv_id"])
        if current is None:
            rows[record["arxiv_id"]] = dict(record)
            continue
        versions = sorted(set(current["versions"]) | set(record["versions"]))
        if record["version"] >= current["version"]:
            merged = dict(record)
            merged["published"] = current["published"]
            merged["listing_date"] = current["listing_date"]
        else:
            merged = dict(current)
        merged["versions"] = versions
        rows[record["arxiv_id"]] = merged
    out = pd.DataFrame(list(rows.values()), columns=COLUMNS)
    return out.sort_values("arxiv_id", ignore_index=True)
