"""Stratified sample of papers for a person to label against a profile.

    python -m rag_papers.labeling sample frontier-models --n 200 --days 30

Why stratified: maybe one paper in ten matches a profile, so a plain random sample of 200 would
hold about 20 positives, and recall would have an interval of about ±0.2. The sample therefore
takes the same number of papers from each similarity band (the top 5%, 5 to 20%, 20 to 50% and
the bottom half). Every row keeps its band and weight, and `stats.weighted_report` turns the
labels back into estimates for the whole stream of papers. The bottom band is sampled too, so
a classifier that misses relevant papers far from the profile text is caught.

The labeller sees papers in random order with no band, score or prediction: the file written
here keeps those columns, but the labelling page must not show them.
"""

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rag_papers import profiles as profiles_module
from rag_papers import store
from rag_papers.ingest import DEFAULT_PATH

LABEL_DIR = Path("eval/profiles")
# Upper edges of the similarity bands, as shares of the pool ranked by similarity.
BANDS = (0.05, 0.20, 0.50, 1.00)
COLUMNS = ["arxiv_id", "title", "abstract", "url", "listing_date", "band", "weight", "relevant"]


def pool(df: pd.DataFrame, profile: profiles_module.Profile, start: date, end: date) -> pd.DataFrame:
    """Papers a profile could match: in its categories, listed in the window, embedded, not an example."""
    listed = pd.to_datetime(df["listing_date"]).dt.date
    in_window = (listed >= start) & (listed <= end)
    in_categories = df["categories"].map(lambda cats: bool(set(cats) & set(profile.categories)))
    embedded = df["embedding"].map(lambda v: hasattr(v, "__len__") and len(v) > 0)
    not_example = ~df["arxiv_id"].isin(profile.example_ids)
    return df[in_window & in_categories & embedded & not_example].reset_index(drop=True)


def sample(
    candidates: pd.DataFrame,
    profile_vector: np.ndarray,
    n: int,
    seed: int,
    exclude: set[str] = frozenset(),
) -> pd.DataFrame:
    if candidates.empty:
        raise ValueError("no candidate papers in the window")
    ranked = candidates.assign(similarity=np.vstack(candidates["embedding"].to_numpy()) @ profile_vector)
    ranked = ranked.sort_values("similarity", ascending=False, ignore_index=True)
    size = len(ranked)
    per_band = n // len(BANDS)
    rng = np.random.default_rng(seed)
    picked, lower = [], 0
    for b, share in enumerate(BANDS):
        upper = max(lower + 1, round(size * share)) if b < len(BANDS) - 1 else size
        band = ranked.iloc[lower:upper]
        # Papers already labelled elsewhere (e.g. the 30 used to diagnose the judge) are left out
        # of the draw but still count in the band size, so the weights describe the whole stream.
        eligible = band[~band["arxiv_id"].isin(exclude)]
        take = min(per_band, len(eligible))
        if take:
            chosen = eligible.iloc[np.sort(rng.choice(len(eligible), take, replace=False))]
            picked.append(chosen.assign(band=b, weight=len(band) / take))
        lower = upper
    out = pd.concat(picked, ignore_index=True)
    # Random order, so the labeller cannot guess a paper's band from its position.
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    out["relevant"] = ""
    out["weight"] = out["weight"].round(4)
    return out.reindex(columns=COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["sample"])
    parser.add_argument("profile", help="profile slug, e.g. frontier-models")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude", type=Path, help="CSV with an arxiv_id column of papers to leave out")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from rag_papers.embeddings import OnnxEmbedder

    profile = profiles_module.load(profiles_module.PROFILE_DIR / f"{args.profile}.md")
    df = store.load(args.path)
    end = pd.to_datetime(df["listing_date"]).dt.date.max()
    candidates = pool(df, profile, end - timedelta(days=args.days), end)
    exclude = set(pd.read_csv(args.exclude, dtype=str)["arxiv_id"]) if args.exclude else set()
    out = sample(candidates, OnnxEmbedder().encode_query(profile.description), args.n, args.seed, exclude)
    target = LABEL_DIR / args.profile / "to_label.csv"
    if target.exists():
        raise SystemExit(f"{target} already exists; a new sample would change what was drawn")
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    logging.info("wrote %s: %d papers from a pool of %d", target, len(out), len(candidates))


if __name__ == "__main__":
    main()
