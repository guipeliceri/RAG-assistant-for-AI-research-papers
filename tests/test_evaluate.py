import json

import numpy as np
import pandas as pd
import pytest

from rag_papers import evaluate
from rag_papers.judge import agreement, cohen_kappa, judge, precision_at_k
from rag_papers.llm import LLMError


class KeywordJudge:
    """Says relevant when 'retrieval' is in the abstract."""

    model = "keyword-judge"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        relevant = "retrieval" in user.lower()
        return f'Sure. {{"relevant": {json.dumps(relevant)}, "reason": "keyword"}}'


def papers(n=60):
    rows = []
    for i in range(n):
        topic = "retrieval" if i % 2 == 0 else "speech"
        v = np.array([1.0, i / n])
        rows.append(
            {
                "arxiv_id": f"2410.{i:05d}",
                "title": f"Paper {i}",
                "abstract": f"About {topic}.",
                "url": f"https://arxiv.org/abs/2410.{i:05d}",
                "embedding": v / np.linalg.norm(v),
            }
        )
    return pd.DataFrame(rows)


def test_judge_parses_json_inside_text():
    assert judge("t", "about retrieval", "profile", KeywordJudge())["relevant"] is True


def test_judge_rejects_non_json():
    class Chatty:
        model = "x"

        def complete(self, system, user):
            return "I think it is relevant!"

    with pytest.raises(LLMError):
        judge("t", "a", "p", Chatty())


def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0


def test_agreement_reports_both_directions_of_disagreement():
    result = agreement([True, False, False], [True, True, False])
    assert result["accuracy"] == pytest.approx(0.667, abs=1e-3)
    assert result["judge_says_relevant_human_not"] == 1
    assert result["human_says_relevant_judge_not"] == 0


def test_precision_at_k():
    assert precision_at_k([True, True, False, True, False, True]) == 0.6
    assert precision_at_k([]) is None


def test_golden_template_is_stratified_and_unlabelled():
    template = evaluate.golden_template(papers(), np.array([1.0, 0.0]))
    assert len(template) == 30
    assert template["arxiv_id"].is_unique
    assert set(template["relevant"]) == {""}
    # The 10 most profile-like papers are always in the sample.
    assert {f"2410.{i:05d}" for i in range(10)} <= set(template["arxiv_id"])


def test_calibrate_uses_only_labelled_rows_and_caches():
    df = papers(4)
    golden = pd.DataFrame(
        {"arxiv_id": [f"2410.{i:05d}" for i in range(4)], "relevant": ["1", "0", "0", ""]}
    )
    llm, cache = KeywordJudge(), {}
    result = evaluate.calibrate(df, golden, "profile", llm, cache)
    assert result["n"] == 3
    assert result["accuracy"] == pytest.approx(0.667, abs=1e-3)  # judge says 2410.00002 is relevant
    evaluate.calibrate(df, golden, "profile", llm, cache)
    assert llm.calls == 3
    # A new profile needs new verdicts.
    evaluate.calibrate(df, golden, "another profile", llm, cache)
    assert llm.calls == 6


def test_evaluate_digests_precision():
    df = papers(10)
    picks = [{"arxiv_id": f"2410.{i:05d}", "title": ""} for i in range(5)]
    digests = [
        {"date": "2024-10-08", "profile_id": evaluate.profile_id("profile"), "papers": picks},
        # Built with another profile (or before profiles were recorded): not comparable.
        {"date": "2024-10-09", "profile_id": evaluate.profile_id("old profile"), "papers": picks},
        {"date": "2024-10-10", "papers": picks},
    ]
    result = evaluate.evaluate_digests(df, digests, "profile", KeywordJudge(), {})
    assert result["per_day"] == {"2024-10-08": 0.6}
    assert result["mean_precision_at_5"] == 0.6


def test_update_readme_replaces_only_the_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- metrics:start -->\nold\n<!-- metrics:end -->\noutro\n")
    evaluate.update_readme({"digests": {"mean_precision_at_5": 0.8, "days": 3}}, readme)
    text = readme.read_text()
    assert "old" not in text and "0.80" in text and text.startswith("intro") and text.endswith("outro\n")
