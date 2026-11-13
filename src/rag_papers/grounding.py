"""Per-answer quality checks shown under every chat answer.

Deterministic (free, always on):
- citation coverage: share of sentences that cite a paper;
- valid citations: every cited ID was among the retrieved papers;
- supported numbers: every number appears in a retrieved abstract or the question.

LLM check (optional, one extra request on the user's key):
- supported sentences: each cited sentence is checked against the abstracts it cites only,
  not against all sources, so a correct fact with the wrong citation still fails.

Caveat shown in the app: when the same model answers and checks, it can share its own blind
spots. The offline evaluation (chat_eval) uses a different model and human-labelled questions.
"""

import json
import re
from dataclasses import dataclass, field

from rag_papers.llm import LLMError
from rag_papers.validate import CITATION_RE, Validation

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9\"'(\[])")
MIN_SENTENCE_CHARS = 20

JUDGE_SYSTEM = """You check whether sentences are supported by the paper abstracts they cite.
For each numbered sentence you get the cited abstracts. A sentence is SUPPORTED only if every
fact in it is stated or directly implied by those abstracts. Missing details, overstatements and
facts from elsewhere make it UNSUPPORTED.
Reply with JSON only: [{"n": 1, "supported": true or false, "why": "a few words"}]"""


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[str]
    supported: bool | None = None
    why: str = ""


@dataclass
class AnswerMetrics:
    sentences: int
    cited_sentences: int
    validation: Validation
    checks: list[SentenceCheck] = field(default_factory=list)
    judged: bool = False

    @property
    def citation_coverage(self) -> float | None:
        return self.cited_sentences / self.sentences if self.sentences else None

    @property
    def supported_share(self) -> float | None:
        verdicts = [c.supported for c in self.checks if c.supported is not None]
        return sum(verdicts) / len(verdicts) if verdicts else None


def sentences(answer: str) -> list[str]:
    text = re.sub(r"^\s*[-*•]\s+", "", answer, flags=re.M)
    parts = [p.strip() for line in text.splitlines() for p in SENTENCE_RE.split(line)]
    return [p for p in parts if len(CITATION_RE.sub("", p).strip()) >= MIN_SENTENCE_CHARS]


def _without_citations(sentence: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", CITATION_RE.sub("", sentence)).strip()


def measure(answer: str, validation: Validation) -> AnswerMetrics:
    found = sentences(answer)
    checks = [SentenceCheck(s, list(dict.fromkeys(CITATION_RE.findall(s)))) for s in found]
    return AnswerMetrics(
        sentences=len(found),
        cited_sentences=sum(bool(c.cited) for c in checks),
        validation=validation,
        checks=checks,
    )


def judge(metrics: AnswerMetrics, abstracts: dict[str, str], llm) -> AnswerMetrics:
    """Fill `supported` for each cited sentence whose citations were retrieved. Raises LLMError."""
    todo = [(i, c) for i, c in enumerate(metrics.checks) if c.cited and all(x in abstracts for x in c.cited)]
    if not todo:
        metrics.judged = True
        return metrics
    blocks = []
    for n, (_, check) in enumerate(todo, start=1):
        cited = "\n".join(f"[{x}] {abstracts[x]}" for x in check.cited)
        blocks.append(f"Sentence {n}: {_without_citations(check.sentence)}\nCited abstracts:\n{cited}")
    reply = llm.complete(JUDGE_SYSTEM, "\n\n".join(blocks))
    match = re.search(r"\[.*\]", reply, re.S)
    try:
        items = json.loads(match.group(0) if match else reply)
        verdicts = {int(item["n"]): item for item in items}
    except (ValueError, KeyError, TypeError):
        raise LLMError("sentence check reply was not the expected JSON") from None
    for n, (_, check) in enumerate(todo, start=1):
        if n in verdicts:
            check.supported = bool(verdicts[n].get("supported"))
            check.why = str(verdicts[n].get("why", ""))[:200]
    metrics.judged = True
    return metrics
