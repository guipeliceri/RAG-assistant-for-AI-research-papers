from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rag_papers import labeling, profiles

GOOD = """---
name: Test profile
categories: [cs.CL, cs.LG]
maintainer: someone
---
## Description
Papers about retrieval-augmented generation and search for language models.

## Includes
- rerankers
- hybrid search

## Excludes
- speech recognition

## Examples
relevant: 2410.00001, 2410.00002
not_relevant: 2410.00003
"""


def test_parse_good_profile():
    p = profiles.parse(GOOD, "test-profile")
    assert p.name == "Test profile"
    assert p.categories == ("cs.CL", "cs.LG")
    assert p.includes == ("rerankers", "hybrid search")
    assert p.excludes == ("speech recognition",)
    assert p.example_ids == {"2410.00001", "2410.00002", "2410.00003"}
    assert "Does NOT count" in p.as_prompt()


def test_fingerprint_tracks_content():
    a = profiles.parse(GOOD, "ab")
    b = profiles.parse(GOOD.replace("- rerankers", "- rerankers\n- query rewriting"), "ab")
    assert a.fingerprint != b.fingerprint


@pytest.mark.parametrize(
    "text, slug, message",
    [
        (GOOD, "Bad Name", "file name"),
        (GOOD.replace("---\nname", "name", 1), "x1", "front matter"),
        (GOOD.replace("name: Test profile\n", ""), "x1", "name"),
        (GOOD.replace("categories: [cs.CL, cs.LG]", "categories: cs.CL"), "x1", "categories"),
        (GOOD.replace("cs.LG", "not a category!"), "x1", "categories"),
        (GOOD.replace("- rerankers", "rerankers"), "x1", "bullet"),
        (GOOD.replace("## Excludes", "## Secrets"), "x1", "unknown sections"),
        (GOOD.replace("2410.00003", "2410.00001"), "x1", "both"),
        (GOOD.replace("2410.00003", "2410.00003v2"), "x1", "arXiv IDs"),
        (GOOD + "x" * 7000, "x1", "longer"),
    ],
)
def test_parse_rejects(text, slug, message):
    with pytest.raises(profiles.ProfileError, match=message):
        profiles.parse(text, slug)


def test_repository_profiles_are_valid():
    root = Path(__file__).parent.parent / "profiles"
    loaded = profiles.load_all(root)
    assert loaded and all(p.categories for p in loaded)


def papers(n=400):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        v = rng.normal(size=4)
        rows.append(
            {
                "arxiv_id": f"2410.{i:05d}",
                "title": f"T{i}",
                "abstract": "a",
                "url": "u",
                "categories": ["cs.CL"] if i % 10 else ["cs.CV"],
                "listing_date": date(2024, 10, 1 + i % 20),
                "embedding": v / np.linalg.norm(v),
            }
        )
    return pd.DataFrame(rows)


def test_pool_filters_categories_window_and_examples():
    p = profiles.parse(GOOD.replace("2410.00001", "2410.00011"), "t1")
    out = labeling.pool(papers(), p, date(2024, 10, 1), date(2024, 10, 10))
    assert set(out["categories"].map(tuple)) <= {("cs.CL",)}
    assert "2410.00011" not in set(out["arxiv_id"])
    assert pd.to_datetime(out["listing_date"]).dt.date.max() <= date(2024, 10, 10)


def test_sample_weights_add_up_to_the_pool():
    df = papers()
    exclude = {df["arxiv_id"].iloc[0]}
    out = labeling.sample(df, np.array([1.0, 0, 0, 0]), n=80, seed=1, exclude=exclude)
    assert len(out) == 80
    assert out["band"].value_counts().to_dict() == {0: 20, 1: 20, 2: 20, 3: 20}
    # Each band's weights sum to the band size, so all weights sum to the pool size.
    assert out["weight"].sum() == pytest.approx(len(df), rel=1e-3)
    assert exclude.isdisjoint(out["arxiv_id"])
    assert (out["relevant"] == "").all()
    # Same seed, same draw.
    again = labeling.sample(df, np.array([1.0, 0, 0, 0]), n=80, seed=1, exclude=exclude)
    assert list(out["arxiv_id"]) == list(again["arxiv_id"])
