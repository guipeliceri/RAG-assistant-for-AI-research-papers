"""Automatic evaluation.

    python -m rag_papers.evaluate golden-template   # sample 30 papers for you to label
    python -m rag_papers.evaluate calibrate         # judge vs your labels in eval/golden.csv
    python -m rag_papers.evaluate digests           # precision@5 of every digest
    python -m rag_papers.evaluate faithfulness      # invalid answers on the fixed questions
    python -m rag_papers.evaluate readme            # refresh the metrics table in README.md

Results go to eval/metrics.json. Judge verdicts are cached in eval/judgments.json so a
paper is judged once.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rag_papers import data, store
from rag_papers.digest import DIGEST_DIR, PROFILE_PATH
from rag_papers.ingest import DEFAULT_PATH
from rag_papers.judge import JUDGE_MODEL, agreement, judge, precision_at_k
from rag_papers.llm import GroqLLM, LLM, LLMError

EVAL_DIR = Path("eval")
GOLDEN = EVAL_DIR / "golden.csv"
JUDGMENTS = EVAL_DIR / "judgments.json"
METRICS = EVAL_DIR / "metrics.json"
README = Path("README.md")
GOLDEN_SIZE = 30

FAITHFULNESS_QUESTIONS = [
    "What came out recently about retrieval-augmented generation?",
    "Which papers propose new rerankers?",
    "What is new in hybrid search combining sparse and dense retrieval?",
    "How are RAG systems being evaluated?",
    "Which papers study hallucination in retrieval-augmented models?",
    "What came out about LLM agents that use tools?",
    "Are there new benchmarks for long-context question answering?",
    "What papers discuss chunking strategies for documents?",
    "Which papers compare long-context models with retrieval?",
    "What is new in query rewriting or query expansion?",
    "Which papers fine-tune embedding models for retrieval?",
    "What came out about multilingual retrieval?",
    "Which papers report results on BEIR or MS MARCO?",
    "What is new in retrieval for code?",
    "Which papers use knowledge graphs together with retrieval?",
]


def _load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def judge_llm() -> LLM:
    return GroqLLM(os.environ["GROQ_API_KEY"], model=os.environ.get("JUDGE_MODEL", JUDGE_MODEL))


def golden_template(df: pd.DataFrame, profile_vector: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """30 papers to label: 10 most similar to the profile, 10 from the middle, 10 random.

    Stratifying makes sure the sample has both relevant and irrelevant papers.
    """
    df = df.assign(similarity=np.vstack(df["embedding"].to_numpy()) @ profile_vector)
    df = df.sort_values("similarity", ascending=False, ignore_index=True)
    third = GOLDEN_SIZE // 3
    top = df.head(third)
    middle_start = len(df) // 2 - third // 2
    middle = df.iloc[middle_start : middle_start + third]
    rest = df.drop(top.index).drop(middle.index)
    sample = pd.concat([top, middle, rest.sample(GOLDEN_SIZE - 2 * third, random_state=seed)])
    # Shuffle so the labeller cannot tell which stratum a paper came from.
    sample = sample.sample(frac=1, random_state=seed)
    return pd.DataFrame(
        {"arxiv_id": sample["arxiv_id"], "title": sample["title"], "url": sample["url"], "relevant": ""}
    )


def judged(paper_id: str, title: str, abstract: str, profile: str, llm: LLM, cache: dict) -> bool:
    if paper_id not in cache:
        cache[paper_id] = judge(title, abstract, profile, llm)
    return cache[paper_id]["relevant"]


def calibrate(df: pd.DataFrame, golden: pd.DataFrame, profile: str, llm: LLM, cache: dict) -> dict:
    labelled = golden[golden["relevant"].astype(str).str.strip().isin(["0", "1"])]
    papers = df.set_index("arxiv_id")
    human, verdicts = [], []
    for _, row in labelled.iterrows():
        paper = papers.loc[row["arxiv_id"]]
        human.append(str(row["relevant"]).strip() == "1")
        verdicts.append(judged(row["arxiv_id"], paper["title"], paper["abstract"], profile, llm, cache))
    return agreement(human, verdicts)


def evaluate_digests(df: pd.DataFrame, digests: list[dict], profile: str, llm: LLM, cache: dict) -> dict:
    papers = df.set_index("arxiv_id")
    per_day = {}
    for digest in digests:
        labels = []
        for p in digest["papers"]:
            abstract = papers.at[p["arxiv_id"], "abstract"] if p["arxiv_id"] in papers.index else ""
            labels.append(judged(p["arxiv_id"], p["title"], abstract, profile, llm, cache))
        if labels:
            per_day[digest["date"]] = precision_at_k(labels)
    values = list(per_day.values())
    return {
        "per_day": per_day,
        "mean_precision_at_5": round(sum(values) / len(values), 3) if values else None,
        "days": len(values),
    }


def faithfulness(index, llm: LLM) -> dict:
    from rag_papers.chat import ask

    invalid, failed = [], 0
    for question in FAITHFULNESS_QUESTIONS:
        try:
            answer = ask(question, index, llm)
        except LLMError:
            failed += 1
            continue
        if answer.sources and not answer.validation.ok:
            invalid.append(question)
    answered = len(FAITHFULNESS_QUESTIONS) - failed
    return {
        "questions": len(FAITHFULNESS_QUESTIONS),
        "answered": answered,
        "invalid_rate": round(len(invalid) / answered, 3) if answered else None,
        "invalid_questions": invalid,
    }


def _fmt(value, pct=False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if pct else f"{value:.2f}"


def metrics_table(metrics: dict) -> str:
    d = metrics.get("digests", {})
    c = metrics.get("calibration", {})
    f = metrics.get("faithfulness", {})
    rows = [
        ("Digest precision@5 (mean, target ≥ 0.80)", _fmt(d.get("mean_precision_at_5")), f"{d.get('days', 0)} days"),
        ("Judge vs human accuracy", _fmt(c.get("accuracy"), pct=True), f"{c.get('n', 0)} labelled papers"),
        ("Judge vs human Cohen's κ", _fmt(c.get("cohen_kappa")), ""),
        ("Chat answers failing validation", _fmt(f.get("invalid_rate"), pct=True), f"{f.get('answered', 0)} fixed questions"),
    ]
    lines = ["| Metric | Value | Based on |", "| :--- | :--- | :--- |"]
    lines += [f"| {name} | {value} | {basis} |" for name, value, basis in rows]
    return "\n".join(lines)


def update_readme(metrics: dict, readme: Path = README) -> None:
    text = readme.read_text()
    block = f"<!-- metrics:start -->\n{metrics_table(metrics)}\n<!-- metrics:end -->"
    readme.write_text(re.sub(r"<!-- metrics:start -->.*?<!-- metrics:end -->", block, text, flags=re.S))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["golden-template", "calibrate", "digests", "faithfulness", "readme"])
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    metrics = _load_json(METRICS, {})
    if args.action == "readme":
        update_readme(metrics)
        return

    profile = PROFILE_PATH.read_text()
    df = store.load(args.path)

    if args.action == "golden-template":
        from rag_papers.embeddings import OnnxEmbedder

        if GOLDEN.exists():
            raise SystemExit(f"{GOLDEN} already exists; delete it first to resample")
        embedded = data.load_papers(args.path)
        template = golden_template(embedded, OnnxEmbedder().encode_query(profile))
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        template.to_csv(GOLDEN, index=False)
        logging.info("wrote %s: fill the relevant column with 1 or 0", GOLDEN)
        return

    cache = _load_json(JUDGMENTS, {})
    llm = judge_llm()
    try:
        if args.action == "calibrate":
            metrics["calibration"] = calibrate(df, pd.read_csv(GOLDEN, dtype=str).fillna(""), profile, llm, cache)
        elif args.action == "digests":
            digests = [json.loads(p.read_text()) for p in sorted(DIGEST_DIR.glob("*.json"))]
            metrics["digests"] = evaluate_digests(df, digests, profile, llm, cache)
        elif args.action == "faithfulness":
            from rag_papers.embeddings import OnnxEmbedder
            from rag_papers.llm import from_env
            from rag_papers.retrieval import Index

            chat_llm = from_env()
            if chat_llm is None:
                raise SystemExit("GROQ_API_KEY is not set: cannot run the faithfulness questions")
            index = Index(data.load_papers(args.path), OnnxEmbedder())
            metrics["faithfulness"] = faithfulness(index, chat_llm)
    finally:
        _save_json(JUDGMENTS, cache)
    metrics["judge_model"] = llm.model
    _save_json(METRICS, metrics)
    logging.info("%s: %s", args.action, json.dumps(metrics.get(args.action, metrics), ensure_ascii=False))


if __name__ == "__main__":
    main()
