"""Every paper of a day that matches a profile, not just the top 5.

    python -m rag_papers.classify [--day YYYY-MM-DD] [--profile SLUG]

Two stages:

1. **Prefilter (no LLM).** Papers outside the profile's arXiv categories are dropped. If the
   profile has a similarity threshold, chosen on the dev labels for high recall, papers below it
   are dropped too. Without one, every paper in the categories goes to stage 2.
2. **Classifier (LLM).** Each remaining paper is judged against the profile's description,
   includes and excludes. Borderline papers are included by default, because recall matters
   more than precision for this tab; that choice is a setting checked on the dev labels.

Results go to digests/profiles/<slug>/<day>.json with the profile fingerprint and model, so a
metric is never shown next to results from another version.
"""

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rag_papers import llm as llm_module
from rag_papers import profiles as profiles_module
from rag_papers import store
from rag_papers.dates import recent_listing_dates
from rag_papers.ingest import DEFAULT_PATH

RESULTS_DIR = Path("digests/profiles")
CONFIG_DIR = Path("eval/profiles")
CLASSIFIER_MODEL = "openai/gpt-oss-120b"
REQUEST_PAUSE = 2.5

SYSTEM = """You decide whether a new research paper belongs in a reader's feed, given the reader's profile.

{profile}

Decide from the title and abstract only. A paper is relevant if its main contribution is about
something the profile counts: what it proposes, builds, measures or analyses. A topic mentioned
in passing, used as one tool among many, or only in related work does not make it relevant.
Anything under "Does NOT count" is not relevant even if it also touches a counted topic.
{borderline}

Reply with JSON only:
{{"relevant": true or false, "reason": "one short sentence"}}"""

BORDERLINE = {
    "include": "If it is genuinely borderline, answer relevant: the reader prefers an extra paper to a missed one.",
    "exclude": "If it is genuinely borderline, answer not relevant.",
}


@dataclass(frozen=True)
class Settings:
    """Per-profile choices, tuned on dev labels only and stored in eval/profiles/<slug>/config.json."""

    threshold: float | None = None
    borderline: str = "include"
    model: str = CLASSIFIER_MODEL

    @classmethod
    def load(cls, slug: str, directory: Path = CONFIG_DIR) -> "Settings":
        path = directory / slug / "config.json"
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(
            threshold=raw.get("threshold"),
            borderline=raw.get("borderline", "include"),
            model=raw.get("model", CLASSIFIER_MODEL),
        )


def in_categories(df: pd.DataFrame, profile: profiles_module.Profile) -> pd.Series:
    wanted = set(profile.categories)
    return df["categories"].map(lambda cats: bool(set(cats) & wanted))


def similarity(df: pd.DataFrame, profile_vector: np.ndarray) -> np.ndarray:
    if df.empty:
        return np.array([])
    return np.vstack(df["embedding"].to_numpy()) @ profile_vector


def verdict(title: str, abstract: str, profile: profiles_module.Profile, settings: Settings, llm) -> dict:
    """{"relevant": bool, "reason": str}. Raises LLMError when the reply is not the expected JSON."""
    system = SYSTEM.format(profile=profile.as_prompt(), borderline=BORDERLINE[settings.borderline])
    reply = llm.complete(system, f"Title: {title}\n\nAbstract: {abstract}")
    match = re.search(r"\{.*\}", reply, re.S)
    try:
        parsed = json.loads(match.group(0) if match else reply)
        return {"relevant": bool(parsed["relevant"]), "reason": str(parsed.get("reason", ""))[:300]}
    except (ValueError, KeyError, TypeError):
        raise llm_module.LLMError("classifier reply was not the expected JSON") from None


def cache_key(profile: profiles_module.Profile, settings: Settings, arxiv_id: str) -> str:
    # A verdict holds only for the exact profile text, prompt setting and model that produced it.
    return f"{profile.fingerprint}:{settings.borderline}:{settings.model}:{arxiv_id}"


def classify(
    papers: pd.DataFrame,
    profile: profiles_module.Profile,
    settings: Settings,
    profile_vector: np.ndarray | None,
    llm,
    cache: dict,
    pause: float = 0.0,
) -> pd.DataFrame:
    """Rows of `papers` with `similarity`, `stage` ('category', 'threshold' or 'llm'), `relevant`, `reason`."""
    out = papers.copy()
    out["similarity"] = similarity(out, profile_vector) if profile_vector is not None else np.nan
    out["stage"], out["relevant"], out["reason"] = "llm", False, ""
    out.loc[~in_categories(out, profile), "stage"] = "category"
    if settings.threshold is not None:
        out.loc[(out["stage"] == "llm") & (out["similarity"] < settings.threshold), "stage"] = "threshold"
    for i in out.index[out["stage"] == "llm"]:
        key = cache_key(profile, settings, out.at[i, "arxiv_id"])
        if key not in cache:
            # The free tier limits requests per minute; cached verdicts cost nothing.
            if pause:
                time.sleep(pause)
            try:
                cache[key] = verdict(out.at[i, "title"], out.at[i, "abstract"], profile, settings, llm)
            except llm_module.LLMError as exc:
                logging.warning("no verdict for %s: %s", out.at[i, "arxiv_id"], exc)
                out.at[i, "stage"] = "error"
                continue
        out.at[i, "relevant"] = cache[key]["relevant"]
        out.at[i, "reason"] = cache[key]["reason"]
    return out


def day_result(
    df: pd.DataFrame,
    day: date,
    profile: profiles_module.Profile,
    settings: Settings,
    profile_vector: np.ndarray,
    llm,
    cache: dict,
    now: datetime | None = None,
    pause: float = 0.0,
) -> dict:
    todays = df[pd.to_datetime(df["listing_date"]).dt.date == day]
    judged = classify(todays, profile, settings, profile_vector, llm, cache, pause)
    matches = judged[judged["relevant"]].sort_values("similarity", ascending=False)
    return {
        "profile": profile.slug,
        "profile_name": profile.name,
        "profile_fingerprint": profile.fingerprint,
        "model": settings.model,
        "borderline": settings.borderline,
        "threshold": settings.threshold,
        "date": day.isoformat(),
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "candidates": int(len(todays)),
        "classified": int(judged["stage"].isin(["llm", "error"]).sum()),
        "errors": int((judged["stage"] == "error").sum()),
        "papers": [
            {
                "arxiv_id": row["arxiv_id"],
                "title": row["title"],
                "authors": list(row["authors"]),
                "url": row["url"],
                "categories": list(row["categories"]),
                "reason": row["reason"],
            }
            for _, row in matches.iterrows()
        ],
    }


def load_cache(slug: str) -> dict:
    path = CONFIG_DIR / slug / "verdicts.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_cache(slug: str, cache: dict) -> None:
    path = CONFIG_DIR / slug / "verdicts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day", type=date.fromisoformat)
    parser.add_argument("--profile", help="only this profile slug")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from rag_papers.embeddings import OnnxEmbedder

    all_profiles = profiles_module.load_all()
    if args.profile:
        all_profiles = [p for p in all_profiles if p.slug == args.profile]
    df = store.load(args.path)
    embedder = OnnxEmbedder()
    days = [args.day] if args.day else recent_listing_dates(datetime.now(timezone.utc))
    failed = False
    for profile in all_profiles:
        settings = Settings.load(profile.slug)
        llm = llm_module.from_env(settings.model)
        if llm is None:
            raise SystemExit("GROQ_API_KEY is not set")
        cache = load_cache(profile.slug)
        vector = embedder.encode_query(profile.description)
        for day in days:
            target = RESULTS_DIR / profile.slug / f"{day}.json"
            if target.exists():
                old = json.loads(target.read_text())
                if old["profile_fingerprint"] == profile.fingerprint and not old["errors"] and old["candidates"]:
                    continue
            result = day_result(df, day, profile, settings, vector, llm, cache, pause=REQUEST_PAUSE)
            save_cache(profile.slug, cache)
            if not result["candidates"]:
                logging.warning("%s: no papers listed on %s yet", profile.slug, day)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            failed |= bool(result["errors"])
            logging.info(
                "%s %s: %d of %d papers match (%d errors)",
                profile.slug,
                day,
                len(result["papers"]),
                result["candidates"],
                result["errors"],
            )
    if failed:
        raise SystemExit("some papers got no verdict; they are retried on the next run")


if __name__ == "__main__":
    main()
