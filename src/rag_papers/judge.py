"""LLM-as-judge for digest relevance, plus its agreement with human labels."""

import json
import re

from rag_papers.llm import LLM, LLMError

# A different model from the summariser, so the judge is not grading its own work.
JUDGE_MODEL = "openai/gpt-oss-120b"

RUBRIC = """You judge whether a research paper is relevant to a reader's interest profile.

Profile:
{profile}

A paper is RELEVANT if its main contribution is about a topic in the profile: the method,
system, benchmark or analysis it proposes is about that topic. It is NOT RELEVANT if the
topic is only mentioned in passing, used as one tool among many, or appears only in
related work.

Judge from the title and abstract only.

Reply with JSON only, no other text:
{{"relevant": true or false, "reason": "one short sentence"}}"""


def judge(paper_title: str, abstract: str, profile: str, llm: LLM) -> dict:
    """Return {"relevant": bool, "reason": str}. Raises LLMError on a bad reply."""
    reply = llm.complete(RUBRIC.format(profile=profile.strip()), f"Title: {paper_title}\n\nAbstract: {abstract}")
    match = re.search(r"\{.*\}", reply, re.S)
    try:
        parsed = json.loads(match.group(0) if match else reply)
        return {"relevant": bool(parsed["relevant"]), "reason": str(parsed.get("reason", ""))}
    except (ValueError, KeyError, TypeError):
        raise LLMError("judge reply was not the expected JSON") from None


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement between two binary raters, corrected for chance."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(human: list[bool], judge_labels: list[bool]) -> dict:
    n = len(human)
    disagreements = sum(h != j for h, j in zip(human, judge_labels))
    return {
        "n": n,
        "accuracy": round((n - disagreements) / n, 3) if n else None,
        "cohen_kappa": round(cohen_kappa(human, judge_labels), 3) if n else None,
        # Reported on purpose: a judge that always agrees could hide shared bias.
        "judge_says_relevant_human_not": sum(j and not h for h, j in zip(human, judge_labels)),
        "human_says_relevant_judge_not": sum(h and not j for h, j in zip(human, judge_labels)),
    }


def precision_at_k(labels: list[bool], k: int = 5) -> float | None:
    top = labels[:k]
    return round(sum(top) / len(top), 3) if top else None
