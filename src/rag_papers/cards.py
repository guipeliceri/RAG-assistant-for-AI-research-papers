"""Paper cards for the chat: a catchy 3-sentence summary and the technologies a paper uses.

One LLM call covers all the papers of an answer, to spare the user's quota. Facts that the
model could invent are kept out of its hands or checked afterwards:

- title, authors, link, date and code link come from the metadata, never from the LLM;
- every technology must appear word for word in the abstract, or it is dropped and counted;
- a summary that states a number missing from the abstract is flagged.
"""

import json
import re
from dataclasses import dataclass, field

from rag_papers.llm import LLMError
from rag_papers.validate import NUMBER_RE

SYSTEM = """You write short, engaging cards about research papers for a reader browsing results.
For each paper you get its ID, title and abstract. Return JSON only, a list with one object per paper:
[{"arxiv_id": "...", "summary": "...", "technologies": ["...", "..."]}]
Rules:
- summary: exactly three sentences, plain English, catchy but accurate. Sentence 1 is the hook (what problem and why it matters), sentence 2 is what they did, sentence 3 is the main result or "The abstract does not report results." Use only the abstract.
- Never invent numbers. Only repeat a number that is in the abstract.
- technologies: models, methods, datasets, benchmarks or tools the paper uses or proposes, copied exactly as written in the abstract. At most 6. Empty list if none are named.
- Write in {language}."""


@dataclass
class Card:
    arxiv_id: str
    summary: str = ""
    technologies: list[str] = field(default_factory=list)
    dropped_technologies: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in NUMBER_RE.findall(text)}


def check(card: Card, abstract: str, title: str) -> Card:
    """Drop technologies not found in the abstract or title; flag numbers not found there."""
    source = f"{title} {abstract}"
    lowered = source.lower()
    kept, dropped = [], []
    for tech in card.technologies:
        (kept if tech.strip() and tech.strip().lower() in lowered else dropped).append(tech.strip())
    card.technologies, card.dropped_technologies = kept[:6], dropped
    card.unsupported_numbers = sorted(_numbers(card.summary) - _numbers(source))
    return card


def make_cards(papers: list[dict], llm, language: str = "the language of the user's question") -> dict[str, Card]:
    """{arxiv_id: Card} for papers given as {"arxiv_id", "title", "abstract"}. Raises LLMError."""
    if not papers:
        return {}
    blocks = "\n\n".join(f"ID: {p['arxiv_id']}\nTitle: {p['title']}\nAbstract: {p['abstract']}" for p in papers)
    reply = llm.complete(SYSTEM.replace("{language}", language), blocks)
    match = re.search(r"\[.*\]", reply, re.S)
    try:
        items = json.loads(match.group(0) if match else reply)
    except ValueError:
        raise LLMError("card reply was not JSON") from None
    by_id = {p["arxiv_id"]: p for p in papers}
    cards = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("arxiv_id") not in by_id:
            continue
        paper = by_id[item["arxiv_id"]]
        techs = item.get("technologies") or []
        card = Card(
            arxiv_id=paper["arxiv_id"],
            summary=str(item.get("summary") or "").strip(),
            technologies=[str(t) for t in techs if isinstance(t, str | int | float)],
        )
        cards[card.arxiv_id] = check(card, paper["abstract"], paper["title"])
    return cards
