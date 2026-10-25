import json
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_digest_tab_shows_latest_digest(in_tmp):
    (in_tmp / "digests").mkdir()
    digest = {
        "date": "2024-10-08",
        "candidates": 40,
        "papers": [
            {"title": "Hybrid RAG", "url": "https://arxiv.org/abs/2410.00001", "authors": ["A"], "summary": None}
        ],
    }
    (in_tmp / "digests" / "2024-10-08.json").write_text(json.dumps(digest))
    at = streamlit_testing.AppTest.from_file(APP).run()
    assert not at.exception
    assert any("2024-10-08" in s.value for s in at.subheader)
    assert any("Summary unavailable" in m.value for m in at.markdown)


def test_chat_without_key_asks_for_one(in_tmp):
    at = streamlit_testing.AppTest.from_file(APP).run()
    at.chat_input[0].set_value("what about agents?").run()
    assert not at.exception
    assert any("API key" in w.value for w in at.warning)
