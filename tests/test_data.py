import json

import pandas as pd

from rag_papers import data, store


def test_local_mode_without_dataset(monkeypatch):
    monkeypatch.delenv("HF_DATASET_REPO", raising=False)
    assert data.fetch() == (data.DEFAULT_PATH, data.LOCAL_DIGESTS)


def test_load_papers_skips_rows_without_embedding(tmp_path):
    df = pd.DataFrame(
        [
            {"arxiv_id": "a", "embedding": [0.1, 0.2]},
            {"arxiv_id": "b", "embedding": None},
        ],
        columns=store.COLUMNS,
    )
    path = tmp_path / "papers.parquet"
    store.save(df, path)
    assert list(data.load_papers(path)["arxiv_id"]) == ["a"]


def test_latest_digest(tmp_path):
    assert data.latest_digest(tmp_path) is None
    for day in ["2024-10-07", "2024-10-08"]:
        (tmp_path / f"{day}.json").write_text(json.dumps({"date": day}))
    assert data.latest_digest(tmp_path)["date"] == "2024-10-08"
