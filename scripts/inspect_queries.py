"""Run the fixed test queries in vector, keyword and hybrid mode, for manual inspection.

    PYTHONPATH=src python scripts/inspect_queries.py [data/papers.parquet]
"""

import sys
from pathlib import Path

from rag_papers import store
from rag_papers.embeddings import OnnxEmbedder
from rag_papers.retrieval import Index

# Mix of topical questions and exact terms (model names, acronyms) where
# keyword matching should help.
QUERIES = [
    "retrieval-augmented generation for question answering",
    "hybrid search combining BM25 and dense retrieval",
    "rerankers for passage retrieval",
    "evaluating hallucination in RAG systems",
    "LLM agents that use tools",
    "long-context language models versus retrieval",
    "Llama-3.1",
    "GPT-4o",
    "ColBERT late interaction",
    "LoRA fine-tuning",
]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/papers.parquet")
    df = store.load(path)
    index = Index(df, OnnxEmbedder())
    for query in QUERIES:
        print(f"\n## {query}")
        for mode in ("vector", "keyword", "hybrid"):
            print(f"  [{mode}]")
            for hit in index.search(query, k=3, mode=mode):
                print(f"    {hit['arxiv_id']}  {hit['title'][:90]}")


if __name__ == "__main__":
    main()
