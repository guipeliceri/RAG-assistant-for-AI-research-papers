import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from rag_papers import fulltext, hype, hype_run
from rag_papers.hype import FullText


def paper(i, day, authors=("A",), abstract="We study things.", version=1, updated=None):
    v = np.zeros(4)
    v[i % 4] = 1.0
    return {
        "arxiv_id": f"2611.{i:05d}",
        "title": f"Paper {i}",
        "abstract": abstract,
        "authors": list(authors),
        "categories": ["cs.CL"],
        "listing_date": day,
        "url": f"https://arxiv.org/abs/2611.{i:05d}",
        "embedding": v,
        "version": version,
        "updated": updated or datetime(day.year, day.month, day.day, tzinfo=timezone.utc),
    }


def test_week_bounds():
    assert hype.week_of(date(2026, 11, 12)) == date(2026, 11, 9)
    df = pd.DataFrame([paper(1, date(2026, 11, 9)), paper(2, date(2026, 11, 13)), paper(3, date(2026, 11, 16))])
    assert list(hype.week_papers(df, date(2026, 11, 9))["arxiv_id"]) == ["2611.00001", "2611.00002"]
    assert list(hype.week_papers(df, date(2026, 11, 9), until=date(2026, 11, 10))["arxiv_id"]) == ["2611.00001"]


def test_votes_at_horizon_ignores_later_snapshots():
    snaps = pd.DataFrame(
        [
            {"arxiv_id": "a", "daily_date": date(2026, 11, 2), "fetched_at": datetime(2026, 11, 3, tzinfo=timezone.utc), "upvotes": 5},
            {"arxiv_id": "a", "daily_date": date(2026, 11, 2), "fetched_at": datetime(2026, 11, 9, 21, tzinfo=timezone.utc), "upvotes": 20},
            {"arxiv_id": "a", "daily_date": date(2026, 11, 2), "fetched_at": datetime(2026, 11, 20, tzinfo=timezone.utc), "upvotes": 99},
        ]
    )
    assert hype.votes_at_horizon(snaps)["a"] == 20


def test_answer_key_handles_zero_and_ties():
    papers = pd.DataFrame({"arxiv_id": list("abcdef")})
    votes = pd.Series({"a": 50, "b": 30, "c": 10, "d": 10, "e": 0})
    assert hype.answer_key(papers, votes) == {"a", "b", "c", "d"}
    assert hype.answer_key(papers, pd.Series(dtype=float)) == set()


def test_weekly_top_uses_only_closed_weeks():
    pop = pd.DataFrame(
        [
            {"arxiv_id": "old", "daily_date": date(2026, 10, 20), "upvotes": 10, "authors": ["Ada"]},
            {"arxiv_id": "recent", "daily_date": date(2026, 11, 4), "upvotes": 999, "authors": ["Ada"]},
        ]
    )
    top = hype.weekly_top(pop, 26, before=date(2026, 11, 9))
    # The week of Nov 2 is not closed on Nov 9 (its votes are still coming in).
    assert list(top["arxiv_id"]) == ["old"]
    assert hype.author_record(["ada "], top) == 1
    assert hype.author_record(["Bob"], top) == 0


def test_signals():
    df = pd.DataFrame([paper(0, date(2026, 11, 9), authors=("Ada", "Bo"), abstract="Code at github.com/x/y.")])
    s = hype.signals(df, pd.DataFrame({"authors": [["Ada"]]}), np.array([1.0, 0, 0, 0]))
    row = s.iloc[0]
    assert row["n_authors"] == 2 and row["author_record"] == 1 and row["code_link"] and row["trend"] == 1.0


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if system == hype.STAGE_A_SYSTEM:
            ids = [line[4:] for line in user.splitlines() if line.startswith("ID: ")]
            return json.dumps([{"arxiv_id": i, "novelty": int(i[-1]) % 5 + 1, "result": 3, "openness": 2, "reach": 9, "appeal": 0} for i in ids])
        return json.dumps({"probability": 0.2 if "Paper 4" in user else 0.05, "reason": "r"})


def test_stage_a_clamps_and_caches():
    df = pd.DataFrame([paper(i, date(2026, 11, 9)) for i in range(10)])
    llm, cache = FakeLLM(), {}
    out = hype.stage_a(df, llm, cache)
    assert llm.calls == 2  # batches of 8
    assert out["reach"].max() == 5 and out["appeal"].min() == 1
    hype.stage_a(df, llm, cache)
    assert llm.calls == 2


def test_forecast_and_calibration_pipeline():
    monday = date(2026, 11, 9)
    df = pd.DataFrame([paper(i, monday) for i in range(40)])
    llm = FakeLLM()
    result = hype_run.forecast(df, pd.DataFrame(), monday, monday, None, llm, {}, lambda i: FullText())
    assert result["papers"] == 40 and len(result["shortlist"]) == hype.SHORTLIST
    assert result["shortlist"][0]["title"] == "Paper 4"
    uncal = hype_run.apply_calibration(json.loads(json.dumps(result)), None)
    assert not uncal["calibrated"] and uncal["shortlist"][0]["probability"] == 0.2
    scored = [{"week": f"w{w}", "shortlist": result["shortlist"], "papers": 40, "key": ["2611.00004", "2611.00010"]} for w in range(6)]
    cal = hype_run.fit_calibration(scored)
    assert cal is not None and cal["outside"] > 0
    assert hype_run.fit_calibration(scored[:5]) is None
    calibrated = hype_run.apply_calibration(json.loads(json.dumps(result)), cal)
    assert calibrated["calibrated"] and calibrated["shortlist"][0]["arxiv_id"] == "2611.00004"
    entry = hype_run.score_week(calibrated, {"2611.00004", "2611.00010"})
    assert entry["pick_hit"] and entry["shortlist_recall"] == 0.5 and len(entry["probabilities"]) == 40


def test_calibration_is_monotone():
    raw = list(np.linspace(0, 1, 50))
    hits = [r > 0.5 for r in raw]
    hits[5] = True  # a lucky low score must not make the map go down later
    cal = hype.Calibration.fit(raw, hits, ["w"] * 50)
    values = [cal(r) for r in raw]
    assert all(b >= a for a, b in zip(values, values[1:]))
    with pytest.raises(ValueError):
        hype.Calibration.fit([0.1], [True], ["w"])


def test_backtest_text_is_v1_for_later_revisions():
    as_of = date(2026, 11, 13)
    df = pd.DataFrame(
        [
            paper(1, date(2026, 11, 9), abstract="v2 says accepted at NeurIPS", version=2, updated=datetime(2026, 12, 1, tzinfo=timezone.utc)),
            paper(2, date(2026, 11, 9), abstract="revised in the same week", version=2, updated=datetime(2026, 11, 10, tzinfo=timezone.utc)),
        ]
    )
    asked = []

    def fake_v1(ids):
        asked.extend(ids)
        return {i: {"title": "v1 title", "abstract": "v1 text"} for i in ids}

    out = hype_run.as_of_text(df, as_of, fetch_v1=fake_v1)
    assert asked == ["2611.00001"]
    assert out.loc[0, "abstract"] == "v1 text" and out.loc[1, "abstract"] == "revised in the same week"


HTML = """<html><body>
<div class="ltx_authors"><span class="ltx_role_affiliation">Some University</span>
<span class="ltx_role_affiliation">A Lab</span></div>
<section><h2>1 Introduction</h2><p>We introduce a model.</p><math>x^2</math></section>
<section><h2>2 Method</h2><p>Details.</p></section>
<section><h2>5 Conclusion</h2><p>It works.</p></section>
</body></html>"""


def test_fulltext_parse():
    t = fulltext.parse(HTML)
    assert t.introduction == "We introduce a model."
    assert t.conclusion == "It works."
    assert "Some University" in t.affiliations and "A Lab" in t.affiliations
    assert t.source == "arXiv HTML v1"
    assert fulltext.parse("<html></html>").source == "abstract only"
