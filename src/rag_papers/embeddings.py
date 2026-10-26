"""Sentence embeddings for papers and queries."""

from typing import Protocol

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on short queries, not on the documents.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
ONNX_FILE = "onnx/model.onnx"
MAX_TOKENS = 512
DIM = 384


class Embedder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class OnnxEmbedder:
    """bge-small on CPU through ONNX Runtime, L2-normalised so dot product equals cosine similarity.

    Same weights and pooling (CLS token) as sentence-transformers, without pulling in torch, so
    the app fits on Streamlit Community Cloud. chromadb already depends on onnxruntime and
    tokenizers.
    """

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32, session=None, tokenizer=None):
        if session is None or tokenizer is None:
            import onnxruntime
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            session = onnxruntime.InferenceSession(
                hf_hub_download(model_name, ONNX_FILE), providers=["CPUExecutionProvider"]
            )
            tokenizer = Tokenizer.from_file(hf_hub_download(model_name, "tokenizer.json"))
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        tokenizer.enable_padding()
        self.session = session
        self.tokenizer = tokenizer
        self.inputs = {i.name for i in session.get_inputs()}
        self.batch_size = batch_size

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            encodings = self.tokenizer.encode_batch(texts[i : i + self.batch_size])
            feed = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
            }
            hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            out.append(hidden[:, 0])
        vectors = np.concatenate(out) if out else np.zeros((0, DIM), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(list(texts))

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([QUERY_INSTRUCTION + text])[0]


def document_text(title: str, abstract: str) -> str:
    return f"{title}. {abstract}"
