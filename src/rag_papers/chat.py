"""Question answering over the retrieved abstracts, with mandatory citations."""

from dataclasses import dataclass
from datetime import date

from rag_papers.llm import LLM, LLMError
from rag_papers.retrieval import Index
from rag_papers.validate import Validation, validate

TOP_K = 8

SYSTEM = """You answer questions about recent AI research papers using only the abstracts provided.
Rules:
- Every sentence that states a fact must end with the citation of the paper it comes from, written as [arxiv_id], e.g. [2410.12345].
- Cite only IDs from the provided list. Never cite anything else.
- Never invent numbers. Only use a number if it appears in a provided abstract.
- If the abstracts do not answer the question, say so plainly instead of guessing.
- Answer in the language of the question.
- Be concise."""

# The papers and the embedding model are English-only, so other languages retrieve poorly.
REWRITE_SYSTEM = """Turn the user's question into a short English search query for finding AI research papers.
Translate it if needed. Keep technical terms and model names as written. Leave out dates and time
words ("this week", "recent"), because dates are filtered separately. Output only the query, nothing else."""
MAX_QUERY_CHARS = 300


@dataclass
class Answer:
    text: str
    sources: list[dict]
    validation: Validation
    search_query: str = ""


def _context(hits: list[dict], index: Index) -> str:
    blocks = []
    for hit in hits:
        abstract = index.df.at[hit["arxiv_id"], "abstract"]
        blocks.append(f"[{hit['arxiv_id']}] {hit['title']} (listed {hit['listing_date']})\n{abstract}")
    return "\n\n".join(blocks)


def search_query(question: str, llm: LLM) -> str:
    """English keywords for retrieval; falls back to the question itself if the LLM fails."""
    try:
        query = llm.complete(REWRITE_SYSTEM, question).strip().strip('"').strip()
    except LLMError:
        return question
    return query if 0 < len(query) <= MAX_QUERY_CHARS else question


def ask(
    question: str,
    index: Index,
    llm: LLM,
    start: date | None = None,
    end: date | None = None,
    k: int = TOP_K,
) -> Answer:
    """Rewrite the question for search, retrieve, generate and validate. Raises LLMError if the provider fails."""
    if not index.has_papers(start, end):
        return Answer("No papers in the selected date range.", [], Validation(has_citations=False))
    query = search_query(question, llm)
    hits = index.search(query, start, end, k)
    if not hits:
        return Answer(
            "No papers in the selected date range match this question.", [], Validation(has_citations=False), query
        )
    prompt = f"Abstracts:\n\n{_context(hits, index)}\n\nQuestion: {question}"
    text = llm.complete(SYSTEM, prompt)
    sources = {
        h["arxiv_id"]: f"{h['title']} {h['listing_date']} {index.df.at[h['arxiv_id'], 'abstract']}"
        for h in hits
    }
    return Answer(text, hits, validate(text, sources, question), query)
