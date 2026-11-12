import json
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def html_blocks(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_three_tabs():
    at = streamlit_testing.AppTest.from_file(APP).run()
    assert [t.label for t in at.tabs] == ["Profiles", "Ask the papers", "Hype forecast"]


def test_profiles_tab_shows_latest_digest_as_cards(in_tmp):
    (in_tmp / "digests").mkdir()
    digest = {
        "date": "2024-10-08",
        "generated_at": "2024-10-09T01:40:00+00:00",
        "candidates": 40,
        "papers": [
            {"title": "Hybrid <RAG>", "url": "https://arxiv.org/abs/2410.00001", "authors": ["A"], "summary": None}
        ],
    }
    (in_tmp / "digests" / "2024-10-08.json").write_text(json.dumps(digest))
    at = streamlit_testing.AppTest.from_file(APP).run()
    assert not at.exception
    page = html_blocks(at)
    # Paper text is escaped before it goes into HTML.
    assert "Hybrid &lt;RAG&gt;" in page and "Hybrid <RAG>" not in page
    assert "Summary unavailable" in page
    # 01:40 UTC is 22:40 of the day before in Brasília.
    assert "08/10/2024 at 22:40 (Brasília time)" in page


def test_chat_without_key_asks_for_one(in_tmp):
    at = streamlit_testing.AppTest.from_file(APP).run()
    at.chat_input[0].set_value("what about agents?").run()
    assert not at.exception
    assert any("API key" in w.value for w in at.warning)


def test_hype_tab_shows_this_weeks_answer_key(in_tmp):
    from datetime import date, datetime, timezone

    import pandas as pd

    from rag_papers import popular

    today = date.today().isoformat()
    now = datetime.now(timezone.utc)
    rows = [
        popular.parse({"paper": {"id": i, "title": f"Voted {i}", "upvotes": v, "submittedOnDailyAt": today}}, now)
        for i, v in [("2610.1", 5), ("2610.2", 90)]
    ]
    (in_tmp / "data").mkdir()
    pd.DataFrame(rows).reindex(columns=popular.COLUMNS).to_parquet(in_tmp / "data" / "popular.parquet", index=False)
    at = streamlit_testing.AppTest.from_file(APP).run()
    assert not at.exception
    page = html_blocks(at)
    assert page.index("Voted 2610.2") < page.index("Voted 2610.1")
    assert "▲ 90 votes" in page
    assert "(Brasília time)" in page


def test_hype_tab_without_data(in_tmp):
    at = streamlit_testing.AppTest.from_file(APP).run()
    assert not at.exception
    assert any("No votes collected yet" in i.value for i in at.info)


def test_profiles_tab_lists_every_match(in_tmp):
    from rag_papers import profiles

    profile = profiles.load(Path(APP).parent / "profiles" / "frontier-models.md")
    folder = in_tmp / "digests" / "profiles" / profile.slug
    folder.mkdir(parents=True)
    result = {
        "profile": profile.slug,
        "profile_name": profile.name,
        "profile_fingerprint": profile.fingerprint,
        "model": "m",
        "date": "2024-10-08",
        "generated_at": "2024-10-08T21:05:00+00:00",
        "candidates": 90,
        "classified": 60,
        "errors": 0,
        "papers": [
            {"arxiv_id": f"2410.0000{i}", "title": f"Match {i}", "authors": ["A"], "url": "u",
             "categories": ["cs.CL"], "reason": "about scaling"}
            for i in range(7)
        ],
    }
    (folder / "2024-10-08.json").write_text(json.dumps(result))
    at = streamlit_testing.AppTest.from_file(APP).run()
    assert not at.exception
    page = html_blocks(at)
    assert all(f"Match {i}" in page for i in range(7))
    assert "08/10/2024 at 18:05 (Brasília time)" in page
    assert "legacy ranking" not in page
    assert any("7 of 90 new papers match" in c.value for c in at.caption)


def test_chat_shows_quality_tiles_and_cards(in_tmp, monkeypatch):
    from datetime import date as d

    import pandas as pd

    from rag_papers import chat, embeddings, grounding, llm
    from rag_papers.embed import embed_missing

    from test_retrieval import HashingEmbedder

    rows = [
        {"arxiv_id": "2410.00001", "title": "Hybrid retrieval for RAG", "authors": ["Ada"],
         "abstract": "We fuse BM25 and dense retrieval and gain 12.5% on BEIR.", "categories": ["cs.CL"],
         "listing_date": d.today(), "url": "https://arxiv.org/abs/2410.00001", "embedding": None},
    ]
    df, _ = embed_missing(pd.DataFrame(rows), HashingEmbedder())
    (in_tmp / "data").mkdir()
    df.to_parquet(in_tmp / "data" / "papers.parquet", index=False)
    monkeypatch.setattr(embeddings, "OnnxEmbedder", HashingEmbedder)

    def fake_complete(self, system, user):
        if system == chat.REWRITE_SYSTEM:
            return "hybrid retrieval"
        if system == chat.SYSTEM:
            return "Hybrid search fuses BM25 with dense retrieval [2410.00001]. It gains 12.5% on BEIR [2410.00001]."
        if system == grounding.JUDGE_SYSTEM:
            return '[{"n": 1, "supported": true}, {"n": 2, "supported": true}]'
        return ('[{"arxiv_id": "2410.00001", "summary": "Search is hard. They fuse two retrievers. '
                'They gain 12.5%.", "technologies": ["BM25", "Transformers"]}]')

    monkeypatch.setattr(llm.OpenAICompatibleLLM, "complete", fake_complete)
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=30).run()
    at.text_input(key="api_key").set_value("k").run()
    at.chat_input[0].set_value("anything on hybrid search?").run()
    assert not at.exception
    page = html_blocks(at)
    assert "Sentences with a citation" in page and "100%" in page
    assert "Sentences supported" in page
    assert "They fuse two retrievers." in page
    # A technology that is not in the abstract never reaches the card.
    assert ">BM25<" in page and "Transformers" not in page
    assert "Ada" in page


def test_hype_tab_shows_forecast(in_tmp):
    folder = in_tmp / "digests" / "hype"
    folder.mkdir(parents=True)
    forecast = {
        "week": "2026-11-09", "as_of": "2026-11-12", "model": "m", "papers": 400, "calibrated": False,
        "outside_probability": None, "generated_at": "2026-11-12T21:04:00+00:00",
        "shortlist": [
            {"arxiv_id": "2611.00001", "title": "Big <new> model", "authors": ["Ada"], "url": "u", "abstract": "a",
             "listing_date": "2026-11-10", "raw": 0.31, "probability": 0.31, "reason": "Open weights, strong results.",
             "source": "arXiv HTML v1", "affiliations": "A Lab",
             "signals": {"n_authors": 1, "author_record": 2, "code_link": True, "trend": 0.6},
             "stage_a": {"novelty": 5, "result": 5, "openness": 5, "reach": 5, "appeal": 4}},
        ],
    }
    (folder / "2026-11-09.json").write_text(json.dumps(forecast))
    at = streamlit_testing.AppTest.from_file(APP).run()
    assert not at.exception
    page = html_blocks(at)
    assert "Most likely to be hyped" in page and "Big &lt;new&gt; model" in page
    assert "31% chance of top 3" in page and "not calibrated yet" in page
    assert "12/11/2026 at 18:04 (Brasília time)" in page
