from types import SimpleNamespace

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

from rag_papers.embeddings import QUERY_INSTRUCTION, OnnxEmbedder


def make_tokenizer():
    words = ["[PAD]", "[UNK]", "a", "b", "c"] + QUERY_INSTRUCTION.strip(": ").lower().split()
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


class FakeSession:
    """Hidden state per token = one-hot of its id; the first (CLS-position) token is pooled."""

    def __init__(self):
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, _outputs, feed):
        self.calls.append(feed)
        ids = feed["input_ids"]
        hidden = np.zeros((*ids.shape, 8), dtype=np.float32)
        for (i, j), v in np.ndenumerate(ids):
            hidden[i, j, v % 8] = 3.0
        return [hidden]


def test_documents_are_normalised_cls_vectors_in_batches():
    session = FakeSession()
    emb = OnnxEmbedder(session=session, tokenizer=make_tokenizer(), batch_size=2)
    vectors = emb.encode_documents(["a b", "b", "c a b"])
    assert vectors.shape == (3, 8)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert [v.argmax() for v in vectors] == [2, 3, 4]
    assert len(session.calls) == 2
    # Only the inputs the model declares are fed, padded to the batch's longest text.
    assert set(session.calls[0]) == {"input_ids", "attention_mask"}
    assert session.calls[0]["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_query_gets_instruction_prefix():
    session = FakeSession()
    emb = OnnxEmbedder(session=session, tokenizer=make_tokenizer())
    vector = emb.encode_query("c")
    assert vector.shape == (8,)
    assert session.calls[0]["input_ids"].shape[1] > 1
