"""Question answering over the retrieved abstracts, with mandatory citations."""

from dataclasses import dataclass
from datetime import date

from rag_papers.llm import LLM
from rag_papers.retrieval import Index
from rag_papers.validate import Validation, validate

TOP_K = 8

SYSTEM = """You answer questions about recent AI research papers using only the abstracts provided.
Rules:
- Every sentence that states a fact must end with the citation of the paper it comes from, written as [arxiv_id], e.g. [2410.12345].
- Cite only IDs from the provided list. Never cite anything else.
- Never invent numbers. Only use a number if it appears in a provided abstract.
- If the abstracts do not answer the question, say so plainly instead of guessing.
- Be concise."""


@dataclass
class Answer:
    text: str
    sources: list[dict]
    validation: Validation


def _context(hits: list[dict], index: Index) -> str:
    blocks = []
    for hit in hits:
        abstract = index.df.at[hit["arxiv_id"], "abstract"]
        blocks.append(f"[{hit['arxiv_id']}] {hit['title']} (listed {hit['listing_date']})\n{abstract}")
    return "\n\n".join(blocks)


def ask(
    question: str,
    index: Index,
    llm: LLM,
    start: date | None = None,
    end: date | None = None,
    k: int = TOP_K,
) -> Answer:
    """Retrieve, generate and validate. Raises LLMError if the provider fails."""
    hits = index.search(question, start, end, k)
    if not hits:
        return Answer("No papers in the selected date range match this question.", [], Validation(has_citations=False))
    prompt = f"Abstracts:\n\n{_context(hits, index)}\n\nQuestion: {question}"
    text = llm.complete(SYSTEM, prompt)
    sources = {
        h["arxiv_id"]: f"{h['title']} {h['listing_date']} {index.df.at[h['arxiv_id'], 'abstract']}"
        for h in hits
    }
    return Answer(text, hits, validate(text, sources, question))
