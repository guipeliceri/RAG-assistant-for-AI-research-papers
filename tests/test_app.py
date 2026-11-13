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
