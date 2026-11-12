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
