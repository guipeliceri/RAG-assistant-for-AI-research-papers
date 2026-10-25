"""Hybrid retrieval: in-memory Chroma (dense) + BM25 (keyword), fused with RRF."""

import re
import uuid
from datetime import date

import chromadb
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from rag_papers.embeddings import Embedder

RRF_K = 60
# How many hits each retriever contributes before fusion.
CANDIDATES = 50
CHROMA_BATCH = 4000

# Keeps model names and versions whole: "gpt-4o", "llama-3.1", "8x7b".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal rank fusion of several ranked ID lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class Index:
    def __init__(self, df: pd.DataFrame, embedder: Embedder):
        self.df = df.set_index("arxiv_id", drop=False)
        self.embedder = embedder
        self.ids = list(self.df["arxiv_id"])
        self.ordinals = np.array([d.toordinal() for d in self.df["listing_date"]])
        self.bm25 = BM25Okapi([tokenize(f"{t} {a}") for t, a in zip(df["title"], df["abstract"])])

        self.collection = chromadb.EphemeralClient().create_collection(
            f"papers-{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
        )
        for start in range(0, len(self.ids), CHROMA_BATCH):
            batch = slice(start, start + CHROMA_BATCH)
            self.collection.add(
                ids=self.ids[batch],
                embeddings=[list(e) for e in self.df["embedding"].iloc[batch]],
                metadatas=[{"listing_ordinal": int(o)} for o in self.ordinals[batch]],
            )

    def _in_range(self, start: date | None, end: date | None) -> np.ndarray:
        lo = start.toordinal() if start else -np.inf
        hi = end.toordinal() if end else np.inf
        return (self.ordinals >= lo) & (self.ordinals <= hi)

    def vector_search(self, query: str, mask: np.ndarray, n: int) -> list[str]:
        allowed = int(mask.sum())
        if allowed == 0:
            return []
        lo, hi = int(self.ordinals[mask].min()), int(self.ordinals[mask].max())
        result = self.collection.query(
            query_embeddings=[list(self.embedder.encode_query(query))],
            n_results=min(n, allowed),
            where={"$and": [{"listing_ordinal": {"$gte": lo}}, {"listing_ordinal": {"$lte": hi}}]},
        )
        return result["ids"][0]

    def keyword_search(self, query: str, mask: np.ndarray, n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:n]
        return [self.ids[i] for i in order if np.isfinite(scores[i]) and scores[i] > 0]

    def search(
        self,
        query: str,
        start: date | None = None,
        end: date | None = None,
        k: int = 10,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Top-k papers for `query` whose listing date is within [start, end].

        The date filter is applied before ranking. `mode` is "hybrid",
        "vector" or "keyword" (the last two exist for comparison).
        """
        mask = self._in_range(start, end)
        rankings = []
        if mode in ("hybrid", "vector"):
            rankings.append(self.vector_search(query, mask, CANDIDATES))
        if mode in ("hybrid", "keyword"):
            rankings.append(self.keyword_search(query, mask, CANDIDATES))
        fused = sorted(rrf(rankings).items(), key=lambda item: -item[1])[:k]
        return [
            {
                "arxiv_id": doc_id,
                "title": self.df.at[doc_id, "title"],
                "url": self.df.at[doc_id, "url"],
                "listing_date": self.df.at[doc_id, "listing_date"],
                "score": score,
            }
            for doc_id, score in fused
        ]
