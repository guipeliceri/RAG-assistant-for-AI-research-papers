"""Hype forecast: which of this week's papers will be the most upvoted on Hugging Face Daily Papers.

    python -m rag_papers.hype [--week YYYY-MM-DD]

Definitions (fixed before any result was seen):

- **Week**: the arXiv listings dated Monday to Friday of one ISO week.
- **Universe**: every paper in the store first listed that week (today cs.CL).
- **Answer key**: a paper's upvotes on Daily Papers 7 days after it was featured, read from the
  vote snapshots; a paper never featured has 0. The week's top 3 by that count are the
  positives. The forecast never sees any vote of the week it predicts.
- **Forecast**: for every paper, the probability that it ends in the week's top 3.

Pipeline, run daily at 18:00 Brasília on the papers listed so far in the current week:

1. Signals computed without an LLM (`signals`): number of authors, the authors' past record in
   the weekly top 3 (closed weeks only), a code or project link in the abstract, categories,
   weekday, and how close the paper is to the topics of the last 4 closed weeks' most voted.
2. The agent reads every abstract (stage A, batched) and scores novelty, size of the claimed
   result, openness, reach and title appeal from 1 to 5.
3. The 30 best by stage A get stage B: the agent reads the v1 introduction and conclusion and
   the affiliations from arXiv's HTML, sees the signals, and gives a probability of top 3.
4. That raw probability is mapped through a calibration fitted on earlier closed weeks only
   (`hype_eval`), so "40%" means 40% in past weeks. Before enough weeks exist, the raw value is
   shown and marked "not calibrated".

Text is always v1: a revision can add "accepted at NeurIPS" or a code link after the hype.
"""

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from rag_papers.llm import LLMError

TOP_K = 3
SHORTLIST = 30
STAGE_A_BATCH = 8
TREND_WEEKS = 4
HISTORY_WEEKS = 26
CODE_LINK = re.compile(r"(github\.com|gitlab\.com|huggingface\.co|https?://\S+\.(io|ai|org|com)/)", re.I)


def week_of(day: date) -> date:
    """Monday of the ISO week containing `day`."""
    return day - timedelta(days=day.weekday())


def week_papers(df: pd.DataFrame, monday: date, until: date | None = None) -> pd.DataFrame:
    days = pd.to_datetime(df["listing_date"]).dt.date
    end = monday + timedelta(days=4)
    if until is not None:
        end = min(end, until)
    return df[(days >= monday) & (days <= end)].reset_index(drop=True)


# Answer key ------------------------------------------------------------------------------


def votes_at_horizon(snapshots: pd.DataFrame, horizon_days: int = 7) -> pd.Series:
    """Upvotes of each paper as of `daily_date + horizon` (last snapshot at or before that time)."""
    if snapshots.empty:
        return pd.Series(dtype=float)
    s = snapshots.copy()
    s["fetched_at"] = pd.to_datetime(s["fetched_at"], utc=True)
    cutoff = pd.to_datetime(s["daily_date"]).dt.tz_localize("UTC") + pd.Timedelta(days=horizon_days + 1)
    s = s[s["fetched_at"] < cutoff].sort_values("fetched_at")
    return s.groupby("arxiv_id")["upvotes"].last()


def answer_key(papers: pd.DataFrame, votes: pd.Series, k: int = TOP_K) -> set[str]:
    """IDs of the week's top-k papers by votes at the horizon. Ties at the k-th place are all included."""
    counts = papers["arxiv_id"].map(votes).fillna(0)
    positive = counts[counts > 0]
    if positive.empty:
        return set()
    threshold = positive.sort_values(ascending=False).iloc[min(k, len(positive)) - 1]
    return set(papers.loc[counts.index[counts >= threshold], "arxiv_id"])


def weekly_top(popular: pd.DataFrame, weeks: int, before: date, k: int = TOP_K) -> pd.DataFrame:
    """The top-k papers of each closed week before `before`, from the vote store.

    A week counts as closed once its last day plus 7 days has passed, so its votes have settled.
    """
    if popular.empty:
        return popular
    p = popular.copy()
    p["week"] = pd.to_datetime(p["daily_date"]).dt.date.map(week_of)
    last_closed = week_of(before) - timedelta(days=7 * 2)
    p = p[(p["week"] <= last_closed) & (p["week"] > last_closed - timedelta(days=7 * weeks))]
    return p.sort_values("upvotes", ascending=False).groupby("week").head(k)


# Signals (no LLM) ---------------------------------------------------------------------------


def author_record(authors: list[str], top: pd.DataFrame) -> int:
    """How many past weekly top-3 papers share at least one author with this paper."""
    if top.empty:
        return 0
    names = {a.strip().lower() for a in authors}
    return int(top["authors"].map(lambda xs: bool(names & {x.strip().lower() for x in xs})).sum())


def signals(papers: pd.DataFrame, top_history: pd.DataFrame, trend_vector: np.ndarray | None) -> pd.DataFrame:
    out = pd.DataFrame({"arxiv_id": papers["arxiv_id"]})
    out["n_authors"] = papers["authors"].map(len)
    out["author_record"] = papers["authors"].map(lambda a: author_record(list(a), top_history))
    out["code_link"] = papers["abstract"].map(lambda t: bool(CODE_LINK.search(t or "")))
    out["n_categories"] = papers["categories"].map(len)
    out["weekday"] = pd.to_datetime(papers["listing_date"]).dt.weekday
    if trend_vector is not None and len(papers):
        out["trend"] = np.round(np.vstack(papers["embedding"].to_numpy()) @ trend_vector, 4)
    else:
        out["trend"] = np.nan
    return out


def trend_vector(top_recent: pd.DataFrame, embedder) -> np.ndarray | None:
    """Mean embedding of the recent weekly winners: what has been getting votes lately."""
    if top_recent.empty:
        return None
    texts = (top_recent["title"] + ". " + top_recent["summary"].fillna("")).tolist()
    v = np.asarray(embedder.encode_documents(texts)).mean(axis=0)
    return v / (np.linalg.norm(v) or 1)


# The agent ---------------------------------------------------------------------------------

STAGE_A_SYSTEM = """You are a scout predicting which new AI papers the research community will get excited about
this week (upvotes on Hugging Face Daily Papers). You only see the title and abstract.
Score each paper from 1 (low) to 5 (high) on:
- novelty: a genuinely new model, method or finding, versus an incremental tweak;
- result: how big a result the abstract claims (state of the art, large gains);
- openness: does it say it releases code, weights or data;
- reach: interesting to many researchers and practitioners, versus a niche;
- appeal: is the title catchy and easy to grasp.
Judge only what the text says. Do not guess the authors' fame.
Reply with JSON only: [{"arxiv_id": "...", "novelty": n, "result": n, "openness": n, "reach": n, "appeal": n}]"""

STAGE_B_SYSTEM = """You are a scout predicting which new AI papers will be the most upvoted on Hugging Face
Daily Papers this week. About {n} papers were listed this week; only the top 3 by upvotes count as hits,
so most papers, even good ones, will not be. Base rate: about {base:.1%} of papers end in the top 3.

You get one paper: its title, abstract, the introduction and conclusion when available,
the affiliations, and signals computed from public data available today:
- authors_past_top3: how many past weekly top-3 papers share an author with this one;
- trend: similarity to the topics that won recently (about 0.5 is typical, 0.7+ is very close);
- code_link: whether the text links to code or a project page.

Think about novelty, size of the result, openness, reach and appeal, then give the probability
that this paper ends in this week's top 3.
Reply with JSON only: {{"probability": number between 0 and 1, "reason": "one or two sentences"}}"""

STAGE_A_FIELDS = ("novelty", "result", "openness", "reach", "appeal")


def stage_a(papers: pd.DataFrame, llm, cache: dict) -> pd.DataFrame:
    """1 to 5 scores per paper from its abstract; cached by paper ID and model."""
    todo = [r for _, r in papers.iterrows() if f"{llm.model}:{r['arxiv_id']}" not in cache]
    for start in range(0, len(todo), STAGE_A_BATCH):
        batch = todo[start : start + STAGE_A_BATCH]
        user = "\n\n".join(f"ID: {r['arxiv_id']}\nTitle: {r['title']}\nAbstract: {r['abstract']}" for r in batch)
        reply = llm.complete(STAGE_A_SYSTEM, user)
        match = re.search(r"\[.*\]", reply, re.S)
        try:
            items = json.loads(match.group(0) if match else reply)
        except ValueError:
            raise LLMError("stage A reply was not JSON") from None
        wanted = {r["arxiv_id"] for r in batch}
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and item.get("arxiv_id") in wanted:
                try:
                    scores = {f: min(5, max(1, int(item[f]))) for f in STAGE_A_FIELDS}
                except (KeyError, TypeError, ValueError):
                    continue
                cache[f"{llm.model}:{item['arxiv_id']}"] = scores
    rows = []
    for pid in papers["arxiv_id"]:
        scores = cache.get(f"{llm.model}:{pid}")
        rows.append({"arxiv_id": pid, **(scores or {f: np.nan for f in STAGE_A_FIELDS})})
    out = pd.DataFrame(rows)
    out["stage_a"] = out[list(STAGE_A_FIELDS)].mean(axis=1)
    return out


@dataclass
class FullText:
    introduction: str = ""
    conclusion: str = ""
    affiliations: str = ""
    source: str = "abstract only"


def stage_b(paper: pd.Series, sig: pd.Series, text: FullText, n_week: int, llm) -> dict:
    """{"probability": float, "reason": str} for one shortlisted paper. Raises LLMError."""
    system = STAGE_B_SYSTEM.format(n=n_week, base=TOP_K / max(n_week, 1))
    parts = [
        f"Title: {paper['title']}",
        f"Abstract: {paper['abstract']}",
        f"Affiliations: {text.affiliations or 'not available'}",
        f"Introduction: {text.introduction or 'not available'}",
        f"Conclusion: {text.conclusion or 'not available'}",
        f"Signals: authors_past_top3={int(sig['author_record'])}, trend={sig['trend']}, "
        f"code_link={bool(sig['code_link'])}, n_authors={int(sig['n_authors'])}",
    ]
    reply = llm.complete(system, "\n\n".join(parts))
    match = re.search(r"\{.*\}", reply, re.S)
    try:
        parsed = json.loads(match.group(0) if match else reply)
        p = float(parsed["probability"])
    except (ValueError, KeyError, TypeError):
        raise LLMError("stage B reply was not the expected JSON") from None
    return {"probability": min(1.0, max(0.0, p)), "reason": str(parsed.get("reason", ""))[:400]}


# Calibration -------------------------------------------------------------------------------


@dataclass
class Calibration:
    """Monotone map from raw to calibrated probability, fitted on closed weeks only (isotonic, binned)."""

    edges: list[float]
    values: list[float]
    weeks: list[str]

    def __call__(self, raw: float) -> float:
        i = int(np.searchsorted(self.edges, raw, side="right")) - 1
        return float(self.values[max(0, min(i, len(self.values) - 1))])

    @classmethod
    def fit(cls, raw: list[float], outcome: list[bool], weeks: list[str], bins: int = 5) -> "Calibration":
        r, y = np.asarray(raw, float), np.asarray(outcome, float)
        if len(r) < bins * 5:
            raise ValueError("too few past forecasts to calibrate")
        edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1))[:-1])
        which = np.searchsorted(edges, r, side="right") - 1
        means = [float(y[which == b].mean()) if np.any(which == b) else np.nan for b in range(len(edges))]
        # Pool adjacent violators so the map never lowers a higher raw score.
        values = _pava([m for m in means], [int(np.sum(which == b)) for b in range(len(edges))])
        return cls(edges=[round(float(e), 4) for e in edges], values=[round(v, 4) for v in values], weeks=sorted(set(weeks)))


def _pava(means: list[float], counts: list[int]) -> list[float]:
    blocks = [[m, c, 1] for m, c in zip(means, counts) if c]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m1, c1, n1 = blocks[i]
            m2, c2, n2 = blocks[i + 1]
            blocks[i] = [(m1 * c1 + m2 * c2) / (c1 + c2), c1 + c2, n1 + n2]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for m, _, n in blocks:
        out += [m] * n
    return out
