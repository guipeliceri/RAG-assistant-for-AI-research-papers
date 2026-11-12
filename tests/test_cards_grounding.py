import json

import pytest

from rag_papers import cards, grounding
from rag_papers.llm import LLMError
from rag_papers.validate import validate

PAPER = {
    "arxiv_id": "2410.00001",
    "title": "Hybrid retrieval for RAG",
    "abstract": "We combine BM25 and dense retrieval with a ColBERT reranker and gain 12.5% on BEIR.",
}


class Reply:
    model = "r"

    def __init__(self, text):
        self.text = text
        self.user = None

    def complete(self, system, user):
        self.user = user
        return self.text


def test_cards_keep_only_technologies_in_the_abstract():
    reply = json.dumps(
        [
            {
                "arxiv_id": "2410.00001",
                "summary": "Search is hard. They mix BM25 and dense retrieval. They gain 12.5% and 40% more.",
                "technologies": ["BM25", "colbert reranker", "GPT-4"],
            },
            {"arxiv_id": "9999.99999", "summary": "x", "technologies": []},
        ]
    )
    out = cards.make_cards([PAPER], Reply("Here: " + reply))
    card = out["2410.00001"]
    assert card.technologies == ["BM25", "colbert reranker"]
    assert card.dropped_technologies == ["GPT-4"]
    assert card.unsupported_numbers == ["40"]
    assert set(out) == {"2410.00001"}


def test_cards_raise_on_garbage():
    with pytest.raises(LLMError):
        cards.make_cards([PAPER], Reply("sorry"))
    assert cards.make_cards([], Reply("unused")) == {}


ANSWER = (
    "Hybrid search mixes BM25 and dense vectors [2410.00001]. "
    "It gains 12.5% on BEIR [2410.00001].\n"
    "- Nobody else has tried this before in any paper."
)


def test_measure_counts_cited_sentences():
    m = grounding.measure(ANSWER, validate(ANSWER, {"2410.00001": PAPER["abstract"]}))
    assert m.sentences == 3
    assert m.cited_sentences == 2
    assert m.citation_coverage == pytest.approx(2 / 3)
    assert m.supported_share is None


def test_judge_checks_only_cited_sentences_against_their_own_sources():
    m = grounding.measure(ANSWER, validate(ANSWER, {"2410.00001": PAPER["abstract"]}))
    llm = Reply(json.dumps([{"n": 1, "supported": True, "why": "ok"}, {"n": 2, "supported": False, "why": "no"}]))
    grounding.judge(m, {"2410.00001": PAPER["abstract"]}, llm)
    assert m.judged and m.supported_share == 0.5
    assert "Sentence 3" not in llm.user and "[2410.00001]" in llm.user
    # Citations are stripped from the sentence itself, so the judge cannot lean on them.
    assert "Sentence 1: Hybrid search mixes BM25 and dense vectors.\n" in llm.user


def test_judge_skips_sentences_citing_unretrieved_papers():
    text = "A claim about something important [2410.55555]."
    m = grounding.measure(text, validate(text, {}))
    llm = Reply("unused")
    grounding.judge(m, {}, llm)
    assert llm.user is None and m.supported_share is None
