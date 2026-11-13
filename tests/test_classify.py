import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rag_papers import classify, profile_eval, profiles
from rag_papers.llm import LLMError

PROFILE = profiles.parse(
    """---
name: Retrieval
categories: [cs.CL]
---
## Description
Papers about retrieval-augmented generation and search for language models.

## Excludes
- speech
""",
    "retrieval",
)


class KeywordLLM:
    model = "kw"

    def __init__(self, fail_on=()):
        self.calls = 0
        self.fail_on = fail_on
        self.systems = []

    def complete(self, system, user):
        self.calls += 1
        self.systems.append(system)
        if any(f in user for f in self.fail_on):
            return "not json"
        return json.dumps({"relevant": "retrieval" in user.lower(), "reason": "kw"})


def papers():
    rows = []
    for i, (topic, cats, sim) in enumerate(
        [
            ("retrieval", ["cs.CL"], 0.9),
            ("speech", ["cs.CL"], 0.2),
            ("retrieval", ["cs.CV"], 0.9),
            ("retrieval", ["cs.CL"], 0.3),
        ]
    ):
        rows.append(
            {
                "arxiv_id": f"2410.0000{i}",
                "title": f"{topic} paper {i}",
                "abstract": f"About {topic}.",
                "authors": ["A"],
                "url": f"https://arxiv.org/abs/2410.0000{i}",
                "categories": cats,
                "listing_date": date(2024, 10, 8),
                "embedding": np.array([sim, np.sqrt(1 - sim**2)]),
            }
        )
    return pd.DataFrame(rows)


VECTOR = np.array([1.0, 0.0])


def test_category_filter_and_llm():
    llm = KeywordLLM()
    out = classify.classify(papers(), PROFILE, classify.Settings(), VECTOR, llm, {})
    assert list(out["stage"]) == ["llm", "llm", "category", "llm"]
    assert list(out["relevant"]) == [True, False, False, True]
    assert llm.calls == 3
    assert "Does NOT count" in llm.systems[0] and "prefers an extra paper" in llm.systems[0]


def test_threshold_skips_llm_and_cache_is_reused():
    llm, cache = KeywordLLM(), {}
    settings = classify.Settings(threshold=0.5, borderline="exclude")
    out = classify.classify(papers(), PROFILE, settings, VECTOR, llm, cache)
    assert list(out["stage"]) == ["llm", "threshold", "category", "threshold"]
    assert llm.calls == 1
    classify.classify(papers(), PROFILE, settings, VECTOR, llm, cache)
    assert llm.calls == 1
    # Another borderline rule is another classifier: no reuse.
    classify.classify(papers(), PROFILE, classify.Settings(threshold=0.5), VECTOR, llm, cache)
    assert llm.calls == 2


def test_bad_reply_is_an_error_not_a_no():
    out = classify.classify(papers(), PROFILE, classify.Settings(), VECTOR, KeywordLLM(fail_on=["paper 0"]), {})
    assert out.loc[0, "stage"] == "error" and not out.loc[0, "relevant"]


def test_day_result_lists_every_match():
    r = classify.day_result(papers(), date(2024, 10, 8), PROFILE, classify.Settings(), VECTOR, KeywordLLM(), {})
    assert [p["arxiv_id"] for p in r["papers"]] == ["2410.00000", "2410.00003"]
    assert r["candidates"] == 4 and r["classified"] == 3 and r["errors"] == 0
    assert r["profile_fingerprint"] == PROFILE.fingerprint


def test_verdict_raises_on_garbage():
    with pytest.raises(LLMError):
        classify.verdict("t", "a", PROFILE, classify.Settings(), KeywordLLM(fail_on=["t"]))


def test_threshold_for_recall_uses_weights():
    sim = np.array([0.9, 0.8, 0.7, 0.1])
    truth = np.array([1, 1, 0, 1])
    # The low-similarity positive weighs as much as the other two together.
    w = np.array([1.0, 1.0, 1.0, 2.0])
    assert profile_eval.threshold_for_recall(sim, truth, w, 0.5) == 0.8
    assert profile_eval.threshold_for_recall(sim, truth, w, 0.9) == 0.1
    assert profile_eval.threshold_for_recall(sim, truth, np.ones(4), 0.6) == 0.8


def test_labelled_splits_by_hash(tmp_path):
    d = tmp_path / "x1"
    d.mkdir()
    pd.DataFrame(
        {"arxiv_id": [f"2410.{i:05d}" for i in range(100)], "relevant": ["1", "0", " ", "1"] * 25, "weight": 1, "band": 0}
    ).to_csv(d / "labels.csv", index=False)
    df = profile_eval.labelled("x1", tmp_path)
    assert len(df) == 75
    assert set(df["split"]) == {"dev", "test"}
