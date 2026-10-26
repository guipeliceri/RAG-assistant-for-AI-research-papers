"""Embed every paper in the store that has no embedding yet.

    python -m rag_papers.embed
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from rag_papers import store
from rag_papers.embeddings import Embedder, OnnxEmbedder, document_text
from rag_papers.ingest import DEFAULT_PATH

# Small batches keep peak memory low on an 8 GB machine.
CHUNK_SIZE = 256


def _has_embedding(value) -> bool:
    # Missing values come back from Parquet/pandas as None or NaN.
    return hasattr(value, "__len__") and len(value) > 0


def embed_missing(df: pd.DataFrame, embedder: Embedder) -> tuple[pd.DataFrame, int]:
    """Return `df` with embeddings filled in, and how many were computed."""
    df = df.copy()
    df["embedding"] = df["embedding"].astype(object)
    missing = [i for i, e in enumerate(df["embedding"]) if not _has_embedding(e)]
    for start in range(0, len(missing), CHUNK_SIZE):
        rows = missing[start : start + CHUNK_SIZE]
        texts = [document_text(df.at[i, "title"], df.at[i, "abstract"]) for i in rows]
        for i, vector in zip(rows, embedder.encode_documents(texts)):
            df.at[i, "embedding"] = [float(x) for x in vector]
        logging.info("embedded %d/%d", min(start + CHUNK_SIZE, len(missing)), len(missing))
    return df, len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df, count = embed_missing(store.load(args.path), OnnxEmbedder())
    store.save(df, args.path)
    logging.info("computed %d embeddings, store has %d papers", count, len(df))


if __name__ == "__main__":
    main()
