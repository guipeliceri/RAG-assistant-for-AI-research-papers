"""Precision and recall of a profile's classifier against human labels.

    python -m rag_papers.profile_eval dev  frontier-models   # as often as you like
    python -m rag_papers.profile_eval tune frontier-models   # pick the prefilter threshold on dev
    python -m rag_papers.profile_eval lock frontier-models   # freeze the test labels (once)
    python -m rag_papers.profile_eval test frontier-models   # the number that gets published

Labels live in eval/profiles/<slug>/labels.csv (the sample from `labeling`, with `relevant`
filled with 1 or 0). Each labelled paper is dev or test by a hash of its ID. Everything that is
adjusted (profile text, includes and excludes, borderline rule, threshold) is adjusted while
looking at dev only. Test is evaluated through `test`, which refuses to run if the test labels
changed after the lock and writes every run to the ledger.

Every report has two baselines next to the classifier, so the gain is visible:
- "always yes": precision equals the share of relevant papers, recall is 1.
- "similarity only": the embedding threshold alone, with no LLM, set on dev to the same recall
  target.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rag_papers import classify, splits, stats, store
from rag_papers import llm as llm_module
from rag_papers import profiles as profiles_module
from rag_papers.ingest import DEFAULT_PATH

LABEL_DIR = Path("eval/profiles")
RECALL_TARGET = 0.90
PREFILTER_RECALL = 0.98


def labelled(slug: str, directory: Path = LABEL_DIR) -> pd.DataFrame:
    path = directory / slug / "labels.csv"
    if not path.exists():
        raise SystemExit(f"{path} does not exist yet")
    df = pd.read_csv(path, dtype={"arxiv_id": str, "relevant": str})
    df = df[df["relevant"].str.strip().isin(["0", "1"])].copy()
    df["relevant"] = df["relevant"].str.strip() == "1"
    df["split"] = [splits.split_of(i, salt=slug) for i in df["arxiv_id"]]
    return df.reset_index(drop=True)


def threshold_for_recall(similarity: np.ndarray, truth: np.ndarray, weights: np.ndarray, target: float) -> float:
    """Highest similarity threshold whose weighted recall on these labels is at least `target`."""
    positives = truth.astype(bool)
    if not positives.any():
        raise ValueError("no relevant papers in these labels")
    order = np.argsort(-similarity[positives])
    sims, w = similarity[positives][order], weights[positives][order]
    covered = np.cumsum(w) / w.sum()
    return float(sims[np.searchsorted(covered, target - 1e-12)])


def report(df: pd.DataFrame, predicted: np.ndarray) -> dict:
    return stats.weighted_report(df["relevant"], predicted, df["weight"], df["band"])


def evaluate(
    df: pd.DataFrame,
    papers: pd.DataFrame,
    profile: profiles_module.Profile,
    settings: classify.Settings,
    profile_vector: np.ndarray,
    llm,
    cache: dict,
    baseline_threshold: float | None,
) -> dict:
    rows = papers.set_index("arxiv_id").loc[df["arxiv_id"]].reset_index()
    judged = classify.classify(rows, profile, settings, profile_vector, llm, cache)
    if (judged["stage"] == "error").any():
        raise SystemExit(f"{int((judged['stage'] == 'error').sum())} papers got no verdict; run again later")
    predicted = judged["relevant"].to_numpy(bool)
    out = {
        "classifier": report(df, predicted),
        "always_yes": report(df, np.ones(len(df), bool)),
        "unweighted": stats.binary_report(df["relevant"], predicted),
    }
    if baseline_threshold is not None:
        out["similarity_only"] = report(df, judged["similarity"].to_numpy() >= baseline_threshold)
        out["similarity_only"]["threshold"] = round(baseline_threshold, 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["dev", "tune", "lock", "test"])
    parser.add_argument("profile")
    parser.add_argument("--note", default="", help="why, for lock")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    profile = profiles_module.load(profiles_module.PROFILE_DIR / f"{args.profile}.md")
    df = labelled(args.profile)
    # Example IDs are shown to the classifier, so they can never measure it.
    df = df[~df["arxiv_id"].isin(profile.example_ids)]
    dev, test = df[df["split"] == "dev"], df[df["split"] == "test"]
    name = f"profile-{args.profile}"

    if args.action == "lock":
        path = splits.lock(name, dict(zip(test["arxiv_id"], test["relevant"].astype(int))), args.note)
        print(f"locked {len(test)} test labels in {path}")
        return

    from rag_papers.embeddings import OnnxEmbedder

    papers = store.load(args.path)
    vector = OnnxEmbedder().encode_query(profile.description)
    dev_sim = classify.similarity(papers.set_index("arxiv_id").loc[dev["arxiv_id"]].reset_index(), vector)
    dev_truth, dev_w = dev["relevant"].to_numpy(bool), dev["weight"].to_numpy(float)
    settings = classify.Settings.load(args.profile)

    if args.action == "tune":
        threshold = threshold_for_recall(dev_sim, dev_truth, dev_w, PREFILTER_RECALL)
        config = LABEL_DIR / args.profile / "config.json"
        current = json.loads(config.read_text()) if config.exists() else {}
        current.update({"threshold": round(threshold, 4), "threshold_chosen_on": "dev", "prefilter_recall": PREFILTER_RECALL})
        config.write_text(json.dumps(current, indent=2) + "\n")
        print(f"threshold {threshold:.4f} keeps {PREFILTER_RECALL:.0%} of the dev positives (weighted); saved to {config}")
        return

    baseline = threshold_for_recall(dev_sim, dev_truth, dev_w, RECALL_TARGET)
    llm = llm_module.from_env(settings.model)
    if llm is None:
        raise SystemExit("GROQ_API_KEY is not set")
    cache = classify.load_cache(args.profile)
    target = dev if args.action == "dev" else test
    if args.action == "test":
        splits.verify(name, dict(zip(test["arxiv_id"], test["relevant"].astype(int))))
    try:
        result = evaluate(target, papers, profile, settings, vector, llm, cache, baseline)
    finally:
        classify.save_cache(args.profile, cache)
    result.update(
        {
            "split": args.action,
            "profile_fingerprint": profile.fingerprint,
            "model": settings.model,
            "borderline": settings.borderline,
            "threshold": settings.threshold,
            "recall_target": RECALL_TARGET,
        }
    )
    if args.action == "test":
        result["test_runs"] = splits.record_test_run(name, profile.fingerprint, result["classifier"])
    metrics_path = LABEL_DIR / args.profile / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics[args.action] = result
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(result["classifier"], indent=2))


if __name__ == "__main__":
    main()
