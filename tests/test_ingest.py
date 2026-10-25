from datetime import date, datetime, timezone

import pandas as pd

from rag_papers import ingest, store

NOW = datetime(2024, 10, 9, tzinfo=timezone.utc)
SINCE = datetime(2024, 10, 1, tzinfo=timezone.utc)


def test_fetch_stops_at_since(make_client, feed_page):
    records = list(make_client(feed_page).fetch_updated_since("cs.CL", SINCE))
    assert [r["arxiv_id"] for r in records] == ["2410.00002", "2410.00001"]


def test_parse_entry_fields(make_client, feed_page):
    rec = next(make_client(feed_page).fetch_updated_since("cs.CL", SINCE))
    assert rec["version"] == 2
    assert rec["title"] == "Hybrid Retrieval for RAG"
    assert rec["abstract"] == "We combine BM25 and dense retrieval. It works."
    assert rec["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert rec["categories"] == ["cs.CL", "cs.IR"]
    assert rec["primary_category"] == "cs.CL"
    assert rec["url"] == "https://arxiv.org/abs/2410.00002"
    # v1 submitted Tue 2024-10-01 11:30 EDT, before the cutoff.
    assert rec["listing_date"] == date(2024, 10, 2)


def test_running_twice_creates_no_duplicates(tmp_path, make_client, feed_page):
    path = tmp_path / "papers.parquet"
    ingest.run(path, days=30, client=make_client(feed_page), now=NOW)
    ingest.run(path, days=30, client=make_client(feed_page), now=NOW)
    df = store.load(path)
    assert len(df) == 2
    assert df["arxiv_id"].is_unique


def _record(arxiv_id="2410.00003", version=1, title="v1 title", updated="2024-10-07"):
    return {
        "arxiv_id": arxiv_id,
        "version": version,
        "versions": [version],
        "title": title,
        "abstract": "abstract",
        "authors": ["A"],
        "categories": ["cs.CL"],
        "primary_category": "cs.CL",
        "published": pd.Timestamp("2024-10-07T10:00:00Z"),
        "updated": pd.Timestamp(f"{updated}T10:00:00Z"),
        "listing_date": date(2024, 10, 8),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def test_revision_updates_record_in_place(tmp_path):
    path = tmp_path / "papers.parquet"
    store.save(store.upsert(store.load(path), [_record()]), path)
    revised = _record(version=2, title="v2 title", updated="2024-10-20")
    store.save(store.upsert(store.load(path), [revised]), path)

    df = store.load(path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["version"] == 2
    assert row["title"] == "v2 title"
    assert list(row["versions"]) == [1, 2]
    # A revision never moves the paper to a newer digest.
    assert row["listing_date"] == date(2024, 10, 8)


def test_older_version_does_not_overwrite_newer(tmp_path):
    df = store.upsert(store.load(tmp_path / "papers.parquet"), [_record(version=3, title="v3")])
    df = store.upsert(df, [_record(version=1, title="v1")])
    assert df.iloc[0]["title"] == "v3"
    assert list(df.iloc[0]["versions"]) == [1, 3]


def test_fetch_paginates(make_client, feed_page):
    head, rest = feed_page.split("<entry>", 1)
    entries = ["<entry>" + e for e in rest.split("<entry>")]
    entries[-1] = entries[-1].replace("</feed>", "")
    page_one = head + entries[0] + "</feed>"
    page_two = head + entries[1] + entries[2] + "</feed>"

    client = make_client(page_one, page_two)
    records = list(client.fetch_updated_since("cs.CL", SINCE))

    assert [r["arxiv_id"] for r in records] == ["2410.00002", "2410.00001"]
    assert [call["start"] for call in client.session.calls] == [0, 1]
