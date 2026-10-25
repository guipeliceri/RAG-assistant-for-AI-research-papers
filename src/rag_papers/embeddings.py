"""Sentence embeddings for papers and queries."""

from typing import Protocol

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on short queries, not on the documents.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """bge-small on CPU, L2-normalised so dot product equals cosine similarity."""

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(QUERY_INSTRUCTION + text, normalize_embeddings=True)


def document_text(title: str, abstract: str) -> str:
    return f"{title}. {abstract}"
