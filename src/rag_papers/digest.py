"""Daily digest: pick the day's papers closest to the profile, then summarise them.

    python -m rag_papers.digest [--day YYYY-MM-DD]

Selection is embedding similarity only; the LLM never decides what goes in.
"""

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rag_papers import llm as llm_module
from rag_papers import store
from rag_papers.dates import latest_listing_date
from rag_papers.embeddings import Embedder, SentenceTransformerEmbedder
from rag_papers.ingest import DEFAULT_PATH

DIGEST_SIZE = 5
DIGEST_DIR = Path("digests")
PROFILE_PATH = Path("profile.md")

SUMMARY_SYSTEM = """You summarise AI research papers for a daily digest.
Rules:
- Use only the abstract you are given. Do not add facts, results or context from elsewhere.
- Never invent numbers. Only repeat a number if it appears in the abstract.
- If the abstract does not say something (e.g. results, data, or what is compared), write "not stated in the abstract" instead of guessing.
- Two or three sentences, plain English: what the paper does, how, and the main result."""


def select(df: pd.DataFrame, day: date, profile_vector: np.ndarray, n: int = DIGEST_SIZE) -> pd.DataFrame:
    """Top-n papers first listed on `day`, by cosine similarity to the profile."""
    todays = df[df["listing_date"] == day]
    if todays.empty:
        return todays.assign(similarity=[])
    vectors = np.vstack(todays["embedding"].to_numpy())
    similarity = vectors @ profile_vector
    return todays.assign(similarity=similarity).nlargest(n, "similarity")


def summarise(paper: pd.Series, llm: llm_module.LLM | None) -> str | None:
    if llm is None:
        return None
    try:
        return llm.complete(SUMMARY_SYSTEM, f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}")
    except llm_module.LLMError as exc:
        logging.warning("summary failed for %s: %s", paper["arxiv_id"], exc)
        return None


def build(
    df: pd.DataFrame,
    day: date,
    profile: str,
    embedder: Embedder,
    llm: llm_module.LLM | None,
    now: datetime | None = None,
) -> dict:
    picked = select(df, day, embedder.encode_query(profile))
    papers = [
        {
            "arxiv_id": row["arxiv_id"],
            "title": row["title"],
            "authors": list(row["authors"]),
            "url": row["url"],
            "similarity": round(float(row["similarity"]), 4),
            "summary": summarise(row, llm),
        }
        for _, row in picked.iterrows()
    ]
    return {
        "date": day.isoformat(),
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "summary_model": llm.model if llm else None,
        "candidates": int((df["listing_date"] == day).sum()),
        "papers": papers,
    }


def write(digest: dict, directory: Path = DIGEST_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest['date']}.json"
    path.write_text(json.dumps(digest, indent=2, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=date.fromisoformat)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--force", action="store_true", help="rebuild an existing digest")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    day = args.day or latest_listing_date(datetime.now(timezone.utc))
    if (DIGEST_DIR / f"{day}.json").exists() and not args.force:
        logging.info("digest for %s already exists, nothing to do", day)
        return
    llm = llm_module.from_env()
    if llm is None:
        logging.warning("GROQ_API_KEY not set: digest will have titles and links only")
    digest = build(store.load(args.path), day, PROFILE_PATH.read_text(), SentenceTransformerEmbedder(), llm)
    path = write(digest)
    summarised = sum(p["summary"] is not None for p in digest["papers"])
    logging.info("wrote %s: %d papers, %d summarised", path, len(digest["papers"]), summarised)


if __name__ == "__main__":
    main()
