"""Run the hype forecast for a week, and score past weeks against their answer key.

    python -m rag_papers.hype_run forecast            # current week, as of today
    python -m rag_papers.hype_run backtest --weeks 12 # closed weeks, rolling calibration

Backtest rules, so the score is the one a live forecast would have had:
- each week is forecast as of its Monday, with history and trends from weeks closed before it;
- the calibration used for week W is fitted only on weeks before W (rolling origin), and the
  first CALIBRATION_WEEKS weeks are only used for fitting, never scored;
- the answer key uses vote snapshots at 7 days when they exist; older weeks only have the
  latest count, and the report says how many weeks used that fallback;
- the forecasting model's training data ends years before these weeks (gpt-oss: mid 2024),
  so it cannot remember which of these papers became popular.
"""

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rag_papers import hype, stats, store
from rag_papers.embed import document_text
from rag_papers import llm as llm_module
from rag_papers.ingest import DEFAULT_PATH
from rag_papers.popular import DEFAULT_PATH as POPULAR_PATH
from rag_papers.popular import SNAPSHOTS_PATH

OUT_DIR = Path("digests/hype")
EVAL_DIR = Path("eval/hype")
HYPE_MODEL = "openai/gpt-oss-120b"
CALIBRATION_WEEKS = 6
REQUEST_PAUSE = 2.5


def _load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, sort_keys=True) + "\n")


def v1_texts(ids: list[str], session=None) -> dict[str, dict]:
    """Title and abstract of version 1 of each paper, from the arXiv API (batched)."""
    import feedparser
    import requests

    session = session or requests.Session()
    out = {}
    for start in range(0, len(ids), 100):
        batch = ids[start : start + 100]
        response = session.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": ",".join(f"{i}v1" for i in batch), "max_results": len(batch)},
            timeout=60,
        )
        response.raise_for_status()
        for entry in feedparser.parse(response.text).entries:
            pid = entry.id.rsplit("/", 1)[-1].split("v")[0]
            out[pid] = {"title": " ".join(entry.title.split()), "abstract": " ".join(entry.summary.split())}
    return out


def as_of_text(df: pd.DataFrame, as_of: date, embedder=None, fetch_v1=v1_texts) -> pd.DataFrame:
    """Replace the text of papers revised after `as_of` with their v1 text.

    The store keeps the latest version, and a revision posted after the week can mention things
    (acceptance, code release) that a forecast made during the week could not have read.
    """
    updated = pd.to_datetime(df["updated"], utc=True).dt.date
    revised = df[(df["version"].astype(int) > 1) & (updated > as_of)]
    if revised.empty:
        return df
    texts = fetch_v1(list(revised["arxiv_id"]))
    out = df.copy()
    out["embedding"] = out["embedding"].astype(object)
    for i in revised.index:
        t = texts.get(out.at[i, "arxiv_id"])
        if t:
            out.at[i, "title"], out.at[i, "abstract"] = t["title"], t["abstract"]
            if embedder is not None:
                # The embedding is recomputed too, so the trend signal also reads v1.
                out.at[i, "embedding"] = list(embedder.encode_documents([document_text(t["title"], t["abstract"])])[0])
    return out


def forecast(
    df: pd.DataFrame,
    popular: pd.DataFrame,
    monday: date,
    as_of: date,
    embedder,
    llm,
    cache: dict,
    fetch_text,
    pause: float = 0.0,
) -> dict:
    """Raw forecast for the week starting `monday`, using only what was public on `as_of`."""
    import time

    papers = hype.week_papers(df, monday, as_of)
    history = hype.weekly_top(popular, hype.HISTORY_WEEKS, before=monday)
    trend = hype.trend_vector(hype.weekly_top(popular, hype.TREND_WEEKS, before=monday), embedder)
    sig = hype.signals(papers, history, trend)
    a_cache = cache.setdefault("stage_a", {})
    scores = hype.stage_a(papers, llm, a_cache) if len(papers) else pd.DataFrame(columns=["arxiv_id", "stage_a"])
    table = papers.merge(sig, on="arxiv_id").merge(scores, on="arxiv_id")
    shortlist = table.sort_values(["stage_a", "trend"], ascending=False).head(hype.SHORTLIST)
    b_cache = cache.setdefault("stage_b", {})
    rows = []
    for _, row in shortlist.iterrows():
        # One stage B reading per paper per week: the free tier's daily token budget is the limit.
        key = f"{llm.model}:{monday}:{row['arxiv_id']}"
        if key not in b_cache:
            if pause:
                time.sleep(pause)
            text = fetch_text(row["arxiv_id"])
            result = hype.stage_b(row, row, text, len(papers), llm)
            b_cache[key] = {**result, "source": text.source, "affiliations": text.affiliations}
        rows.append(
            {
                "arxiv_id": row["arxiv_id"],
                "title": row["title"],
                "authors": list(row["authors"]),
                "url": row["url"],
                "abstract": row["abstract"],
                "listing_date": str(row["listing_date"]),
                "raw": b_cache[key]["probability"],
                "reason": b_cache[key]["reason"],
                "source": b_cache[key]["source"],
                "affiliations": b_cache[key].get("affiliations", ""),
                "signals": {
                    "n_authors": int(row["n_authors"]),
                    "author_record": int(row["author_record"]),
                    "code_link": bool(row["code_link"]),
                    "trend": None if pd.isna(row["trend"]) else float(row["trend"]),
                },
                "stage_a": {f: (None if pd.isna(row[f]) else int(row[f])) for f in hype.STAGE_A_FIELDS},
            }
        )
    rows.sort(key=lambda r: -r["raw"])
    return {
        "week": monday.isoformat(),
        "as_of": as_of.isoformat(),
        "model": llm.model,
        "papers": int(len(papers)),
        "all_ids": list(papers["arxiv_id"]),
        "shortlist": rows,
    }


def apply_calibration(result: dict, calibration: dict | None) -> dict:
    """Calibrated probability for the shortlist and one shared value for every other paper."""
    if calibration is None:
        result["calibrated"] = False
        for r in result["shortlist"]:
            r["probability"] = r["raw"]
        result["outside_probability"] = None
        return result
    cal = hype.Calibration(calibration["edges"], calibration["values"], calibration["weeks"])
    for r in result["shortlist"]:
        r["probability"] = round(cal(r["raw"]), 4)
    result["calibrated"] = True
    result["calibration_weeks"] = calibration["weeks"]
    result["outside_probability"] = calibration["outside"]
    result["shortlist"].sort(key=lambda r: (-r["probability"], -r["raw"]))
    return result


def fit_calibration(scored_weeks: list[dict]) -> dict | None:
    """Fit on weeks whose answer key is known: shortlist raw -> hit, plus the hit rate outside it."""
    raw, hit, weeks, outside_hits, outside_n = [], [], [], 0, 0
    for w in scored_weeks:
        short = {r["arxiv_id"] for r in w["shortlist"]}
        for r in w["shortlist"]:
            raw.append(r["raw"])
            hit.append(r["arxiv_id"] in w["key"])
            weeks.append(w["week"])
        outside_hits += len(set(w["key"]) - short)
        outside_n += w["papers"] - len(short)
    if len({*weeks}) < CALIBRATION_WEEKS:
        return None
    cal = hype.Calibration.fit(raw, hit, weeks)
    return {
        "edges": cal.edges,
        "values": cal.values,
        "weeks": cal.weeks,
        "outside": round(outside_hits / outside_n, 5) if outside_n else 0.0,
    }


def score_week(result: dict, key: set[str]) -> dict:
    """Per-paper outcomes and probabilities for every paper of the week (outside the shortlist too)."""
    probs = {r["arxiv_id"]: r["probability"] for r in result["shortlist"]}
    outside = result.get("outside_probability") or 0.0
    ids = result["all_ids"]
    p = [probs.get(i, outside) for i in ids]
    y = [i in key for i in ids]
    pick = result["shortlist"][0]["arxiv_id"] if result["shortlist"] else None
    predicted_top = [r["arxiv_id"] for r in result["shortlist"][: hype.TOP_K]]
    return {
        "week": result["week"],
        "key": sorted(key),
        "pick_hit": pick in key if pick else False,
        "top3_overlap": len(set(predicted_top) & key),
        "shortlist_recall": (len(set(probs) & key) / len(key)) if key else None,
        "probabilities": p,
        "outcomes": y,
    }


def baselines(df: pd.DataFrame, popular: pd.DataFrame, monday: date, key: set[str]) -> dict:
    """Two forecasts that need no LLM, scored the same way."""
    papers = hype.week_papers(df, monday)
    history = hype.weekly_top(popular, hype.HISTORY_WEEKS, before=monday)
    record = papers["authors"].map(lambda a: hype.author_record(list(a), history))
    best_author = list(papers.loc[record.sort_values(ascending=False).index[: hype.TOP_K], "arxiv_id"])
    by_authors = list(papers.loc[papers["authors"].map(len).sort_values(ascending=False).index[: hype.TOP_K], "arxiv_id"])
    return {
        "author_record_top3_overlap": len(set(best_author) & key),
        "author_record_pick_hit": bool(best_author and best_author[0] in key),
        "most_authors_top3_overlap": len(set(by_authors) & key),
        "random_expected_top3_overlap": round(hype.TOP_K * len(key) / max(len(papers), 1), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["forecast", "backtest"])
    parser.add_argument("--weeks", type=int, default=12)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from rag_papers import fulltext
    from rag_papers.embeddings import OnnxEmbedder

    llm = llm_module.from_env(HYPE_MODEL)
    if llm is None:
        raise SystemExit("GROQ_API_KEY is not set")
    df = store.load(args.path)
    popular = store.load(POPULAR_PATH) if POPULAR_PATH.exists() else pd.DataFrame()
    snapshots = pd.read_parquet(SNAPSHOTS_PATH) if SNAPSHOTS_PATH.exists() else pd.DataFrame()
    embedder = OnnxEmbedder()
    cache_path = EVAL_DIR / "cache.json"
    cache = _load_json(cache_path, {})
    today = datetime.now(timezone.utc).date()

    try:
        if args.action == "forecast":
            monday = hype.week_of(today)
            result = forecast(df, popular, monday, today, embedder, llm, cache, fulltext.fetch, REQUEST_PAUSE)
            result = apply_calibration(result, _load_json(EVAL_DIR / "calibration.json", None))
            result["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            result.pop("all_ids")
            _save_json(OUT_DIR / f"{monday}.json", result)
            logging.info("forecast for week %s: %d papers, %d shortlisted", monday, result["papers"], len(result["shortlist"]))
            return

        # Backtest: the latest weeks whose answer key has settled (week end + 7 days in the past).
        last = hype.week_of(today) - timedelta(days=14)
        weeks = [last - timedelta(days=7 * i) for i in range(args.weeks)][::-1]
        votes = hype.votes_at_horizon(snapshots)
        final = popular.set_index("arxiv_id")["upvotes"] if len(popular) else pd.Series(dtype=float)
        scored, report, fallback = [], [], 0
        for monday in weeks:
            week_end = monday + timedelta(days=4)
            week_df = as_of_text(hype.week_papers(df, monday), week_end, embedder)
            raw = forecast(week_df, popular, monday, week_end, embedder, llm, cache, fulltext.fetch, REQUEST_PAUSE)
            _save_json(cache_path, cache)
            papers = hype.week_papers(df, monday)
            have = papers["arxiv_id"].isin(votes.index).any()
            fallback += not have
            key = hype.answer_key(papers, votes if have else final)
            calibration = fit_calibration(scored)
            result = apply_calibration(raw, calibration)
            entry = score_week(result, key)
            entry.update({"calibrated": result["calibrated"], "baselines": baselines(df, popular, monday, key)})
            report.append(entry)
            scored.append({"week": raw["week"], "shortlist": raw["shortlist"], "papers": raw["papers"], "key": sorted(key)})
        evaluated = [r for r in report if r["calibrated"]]
        p = [x for r in evaluated for x in r["probabilities"]]
        y = [x for r in evaluated for x in r["outcomes"]]
        summary = {
            "weeks_scored": len(evaluated),
            "weeks_used_only_for_calibration": len(report) - len(evaluated),
            "weeks_with_final_counts_instead_of_7_day_snapshots": fallback,
            "pick_in_top3": stats._proportion(sum(r["pick_hit"] for r in evaluated), len(evaluated)),
            "mean_top3_overlap": round(float(np.mean([r["top3_overlap"] for r in evaluated])), 3) if evaluated else None,
            "shortlist_recall": round(float(np.nanmean([r["shortlist_recall"] or np.nan for r in evaluated])), 3) if evaluated else None,
            "calibration": stats.calibration(y, p) if p else None,
            "baselines": {
                "author_record_mean_top3_overlap": round(float(np.mean([r["baselines"]["author_record_top3_overlap"] for r in evaluated])), 3) if evaluated else None,
                "author_record_pick_in_top3": stats._proportion(sum(r["baselines"]["author_record_pick_hit"] for r in evaluated), len(evaluated)),
                "most_authors_mean_top3_overlap": round(float(np.mean([r["baselines"]["most_authors_top3_overlap"] for r in evaluated])), 3) if evaluated else None,
                "random_expected_top3_overlap": round(float(np.mean([r["baselines"]["random_expected_top3_overlap"] for r in evaluated])), 3) if evaluated else None,
            },
            "model": llm.model,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        for r in report:
            r.pop("probabilities"), r.pop("outcomes")
        _save_json(EVAL_DIR / "backtest.json", {"summary": summary, "weeks": report})
        # The live forecast uses a calibration fitted on every scored week, all before the current one.
        final_cal = fit_calibration(scored)
        if final_cal:
            _save_json(EVAL_DIR / "calibration.json", final_cal)
        logging.info("backtest: %s", json.dumps(summary["pick_in_top3"]))
    finally:
        _save_json(cache_path, cache)


if __name__ == "__main__":
    main()
