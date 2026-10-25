from datetime import date

import pytest

from rag_papers.chat import ask
from rag_papers.llm import LLMError
from rag_papers.retrieval import Index
from rag_papers.validate import citations, validate

from test_retrieval import HashingEmbedder, make_df


class ScriptedLLM:
    model = "scripted"

    def __init__(self, reply):
        self.reply = reply
        self.prompt = None

    def complete(self, system, user):
        self.prompt = user
        return self.reply


@pytest.fixture
def index():
    return Index(make_df(HashingEmbedder()), HashingEmbedder())


SOURCES = {"2410.00001": "Hybrid retrieval. We get 12.5% gains on 3 datasets."}


def test_citations_strip_versions_and_dedupe():
    assert citations("A [2410.00001v2]. B [2410.00001]. C [cs/0101001].") == ["2410.00001", "cs/0101001"]


def test_valid_answer():
    v = validate("Gains of 12.5% on 3 datasets [2410.00001].", SOURCES)
    assert v.ok


def test_flags_citation_not_retrieved():
    v = validate("Something [2410.99999].", SOURCES)
    assert v.invalid_citations == ["2410.99999"]
    assert not v.ok


def test_flags_invented_number():
    v = validate("Gains of 40% [2410.00001].", SOURCES)
    assert v.unsupported_numbers == ["40"]


def test_number_from_question_is_allowed():
    assert validate("The top 5 papers [2410.00001].", SOURCES, question="top 5 papers?").ok


def test_flags_missing_citations():
    v = validate("No sources here.", SOURCES)
    assert not v.has_citations and not v.ok


def test_ask_sends_only_retrieved_abstracts_and_validates(index):
    llm = ScriptedLLM("BM25 and dense retrieval are fused [2410.00001].")
    answer = ask("hybrid retrieval", index, llm, k=2)
    assert "[2410.00001]" in llm.prompt
    assert len(answer.sources) == 2
    assert answer.validation.ok


def test_ask_flags_hallucinated_citation(index):
    answer = ask("hybrid retrieval", index, ScriptedLLM("See [2501.00001]."), k=2)
    assert answer.validation.invalid_citations == ["2501.00001"]


def test_ask_respects_date_range(index):
    llm = ScriptedLLM("x [2410.00004].")
    answer = ask("agents", index, llm, start=date(2024, 10, 8), end=date(2024, 10, 9))
    assert {s["arxiv_id"] for s in answer.sources} <= {"2410.00004", "2410.00005"}


def test_ask_with_no_papers_in_range_does_not_call_llm(index):
    class Exploding:
        model = "x"

        def complete(self, system, user):
            raise AssertionError("should not be called")

    answer = ask("agents", index, Exploding(), start=date(2030, 1, 1), end=date(2030, 1, 2))
    assert answer.sources == []


def test_ask_propagates_provider_errors(index):
    class Broken:
        model = "x"

        def complete(self, system, user):
            raise LLMError("quota exceeded (HTTP 429)")

    with pytest.raises(LLMError):
        ask("agents", index, Broken())
