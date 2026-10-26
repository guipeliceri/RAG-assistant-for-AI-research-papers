import hashlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rag_papers.embed import embed_missing
from rag_papers.retrieval import Index, rrf, tokenize

DIM = 64


class HashingEmbedder:
    """Bag-of-words hashed into a small vector: crude but deterministic."""

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(DIM)
        for token in tokenize(text):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1
        return v / (np.linalg.norm(v) or 1)

    def encode_documents(self, texts):
        return np.array([self._vector(t) for t in texts])

    def encode_query(self, text):
        return self._vector(text)


class BlindEmbedder(HashingEmbedder):
    """Ignores the query: stands in for a dense model that misses rare names."""

    def encode_query(self, text):
        return self._vector("language models")


PAPERS = [
    ("2410.00001", "Hybrid retrieval for RAG", "We fuse BM25 and dense retrieval for question answering.", date(2024, 10, 1)),
    ("2410.00002", "Dense passage retrieval revisited", "Dense retrieval with contrastive training.", date(2024, 10, 2)),
    ("2410.00003", "Evaluating Mixtral-8x7B on reasoning", "We benchmark Mixtral-8x7B against language models.", date(2024, 10, 3)),
    ("2410.00004", "Agents that plan", "Language model agents plan with tools.", date(2024, 10, 8)),
    ("2410.00005", "Language models and retrieval", "Retrieval improves language models.", date(2024, 10, 9)),
]


def make_df(embedder):
    df = pd.DataFrame(
        [
            {"arxiv_id": i, "title": t, "abstract": a, "listing_date": d, "url": f"https://arxiv.org/abs/{i}", "embedding": None}
            for i, t, a, d in PAPERS
        ]
    )
    df, _ = embed_missing(df, embedder)
    return df


@pytest.fixture
def index():
    return Index(make_df(HashingEmbedder()), HashingEmbedder())


def test_tokenize_keeps_model_names_whole():
    assert tokenize("Mixtral-8x7B beats GPT-4o and Llama-3.1.") == [
        "mixtral-8x7b", "beats", "gpt-4o", "and", "llama-3.1",
    ]


def test_rrf_rewards_agreement():
    scores = rrf([["a", "b"], ["b", "c"]])
    assert max(scores, key=scores.get) == "b"


def test_embed_missing_only_fills_gaps():
    df = make_df(HashingEmbedder())
    df.at[0, "embedding"] = None
    _, count = embed_missing(df, HashingEmbedder())
    assert count == 1


@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
def test_date_filter_applies_before_ranking(index, mode):
    hits = index.search("retrieval", start=date(2024, 10, 1), end=date(2024, 10, 2), k=5, mode=mode)
    assert hits
    assert all(date(2024, 10, 1) <= h["listing_date"] <= date(2024, 10, 2) for h in hits)


def test_empty_date_range_returns_nothing(index):
    assert index.search("retrieval", start=date(2025, 1, 1), end=date(2025, 1, 2)) == []


def test_hybrid_finds_exact_model_name_that_vector_misses():
    df = make_df(HashingEmbedder())
    index = Index(df, BlindEmbedder())
    vector_top = index.search("Mixtral-8x7B", k=1, mode="vector")[0]["arxiv_id"]
    hybrid_top = index.search("Mixtral-8x7B", k=1, mode="hybrid")[0]["arxiv_id"]
    assert vector_top != "2410.00003"
    assert hybrid_top == "2410.00003"


class Float32Embedder(HashingEmbedder):
    """Returns float32 vectors, like the ONNX embedder; chromadb rejects lists of np.float32."""

    def encode_documents(self, texts):
        return super().encode_documents(texts).astype(np.float32)

    def encode_query(self, text):
        return super().encode_query(text).astype(np.float32)


def test_float32_embeddings_work_end_to_end():
    embedder = Float32Embedder()
    hits = Index(make_df(embedder), embedder).search("hybrid retrieval", mode="vector", k=2)
    assert hits[0]["arxiv_id"] == "2410.00001"
