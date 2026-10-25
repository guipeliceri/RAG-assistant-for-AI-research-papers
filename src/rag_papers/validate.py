"""Deterministic checks on a generated answer, with no LLM involved."""

import re
from dataclasses import dataclass, field

CITATION_RE = re.compile(r"\[(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\]")
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")


@dataclass
class Validation:
    invalid_citations: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    has_citations: bool = True

    @property
    def ok(self) -> bool:
        return self.has_citations and not self.invalid_citations and not self.unsupported_numbers


def citations(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION_RE.findall(text)))


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in NUMBER_RE.findall(text)}


def validate(answer: str, sources: dict[str, str], question: str = "") -> Validation:
    """Check `answer` against the retrieved `sources` ({arxiv_id: title + abstract}).

    Every cited ID must be one of the sources, and every number in the answer must
    appear in some source (numbers the user typed in the question are allowed).
    """
    cited = citations(answer)
    prose = CITATION_RE.sub(" ", answer)
    allowed = _numbers(" ".join(sources.values())) | _numbers(question)
    return Validation(
        invalid_citations=[c for c in cited if c not in sources],
        unsupported_numbers=sorted(_numbers(prose) - allowed),
        has_citations=bool(cited),
    )
