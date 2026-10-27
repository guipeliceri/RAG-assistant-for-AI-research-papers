import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from rag_papers import digest
from rag_papers.llm import LLMError

DAY = date(2024, 10, 8)


class AxisEmbedder:
    """The profile points along the first axis."""

    def encode_query(self, text):
        return np.array([1.0, 0.0])


class EchoLLM:
    model = "echo"

    def __init__(self):
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append((system, user))
        return "summary of " + user.splitlines()[0]


class BrokenLLM:
    model = "broken"

    def complete(self, system, user):
        raise LLMError("quota exceeded (HTTP 429)")


def paper(arxiv_id, day, vector):
    v = np.array(vector, dtype=float)
    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "abstract": f"Abstract {arxiv_id}",
        "authors": ["A"],
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "listing_date": day,
        "embedding": v / np.linalg.norm(v),
    }


DF = pd.DataFrame(
    [paper(f"2410.0000{i}", DAY, [10 - i, i]) for i in range(7)]
    + [paper("2410.09999", date(2024, 10, 7), [1, 0])]  # perfect match, but another day
)


def test_select_takes_top_n_of_the_day_by_similarity():
    picked = digest.select(DF, DAY, AxisEmbedder().encode_query(""))
    assert list(picked["arxiv_id"]) == [f"2410.0000{i}" for i in range(5)]


def test_select_empty_day():
    assert digest.select(DF, date(2024, 10, 12), np.array([1.0, 0.0])).empty


def test_build_summarises_with_abstract_only():
    llm = EchoLLM()
    result = digest.build(DF, DAY, "profile", AxisEmbedder(), llm, now=datetime(2024, 10, 9, tzinfo=timezone.utc))
    assert result["date"] == "2024-10-08"
    assert result["candidates"] == 7
    assert result["summary_model"] == "echo"
    assert [p["summary"] for p in result["papers"]][0] == "summary of Title: Paper 2410.00000"
    assert "Abstract 2410.00000" in llm.prompts[0][1]
    assert "Never invent numbers" in llm.prompts[0][0]


def test_llm_failure_keeps_titles_and_links():
    result = digest.build(DF, DAY, "profile", AxisEmbedder(), BrokenLLM())
    assert len(result["papers"]) == 5
    assert all(p["summary"] is None for p in result["papers"])
    assert all(p["url"].startswith("https://arxiv.org/abs/") for p in result["papers"])


def test_no_llm_configured():
    result = digest.build(DF, DAY, "profile", AxisEmbedder(), None)
    assert result["summary_model"] is None
    assert all(p["summary"] is None for p in result["papers"])


def test_write_json(tmp_path):
    result = digest.build(DF, DAY, "profile", AxisEmbedder(), None)
    path = digest.write(result, tmp_path)
    assert path.name == "2024-10-08.json"
    assert json.loads(path.read_text())["papers"][0]["arxiv_id"] == "2410.00000"


def run_main(tmp_path, monkeypatch, df, *args):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(digest, "DIGEST_DIR", tmp_path / "digests")
    monkeypatch.setattr(digest, "PROFILE_PATH", tmp_path / "profile.md")
    (tmp_path / "profile.md").write_text("profile")
    monkeypatch.setattr(digest.store, "load", lambda path: df)
    monkeypatch.setattr(digest, "OnnxEmbedder", AxisEmbedder)
    monkeypatch.setattr(digest.llm_module, "from_env", lambda: None)
    monkeypatch.setattr("sys.argv", ["digest", "--day", DAY.isoformat(), *args])
    digest.main()
    return tmp_path / "digests" / f"{DAY}.json"


def test_main_does_not_write_an_empty_digest(tmp_path, monkeypatch):
    assert not run_main(tmp_path, monkeypatch, DF[DF["listing_date"] != DAY]).exists()


def test_main_rebuilds_an_earlier_empty_digest(tmp_path, monkeypatch):
    (tmp_path / "digests").mkdir()
    (tmp_path / "digests" / f"{DAY}.json").write_text(json.dumps({"date": DAY.isoformat(), "papers": []}))
    path = run_main(tmp_path, monkeypatch, DF)
    assert len(json.loads(path.read_text())["papers"]) == 5


def test_build_records_the_profile():
    profile = "# Interest profile\n\nAdvances in frontier models: scaling, reasoning."
    result = digest.build(DF, DAY, profile, AxisEmbedder(), None)
    assert result["profile_name"] == "Advances in frontier models"
    assert result["profile_id"] == digest.profile_id(profile + "\n")
    assert result["profile_id"] != digest.profile_id("something else")
