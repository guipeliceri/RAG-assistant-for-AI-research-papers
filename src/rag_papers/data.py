"""Where the app reads papers and digests from.

On the Space, `HF_DATASET_REPO` is set and everything comes from the Dataset
(the daily job writes there). Locally, the files in the working directory are used.
"""

import json
import os
from pathlib import Path

import pandas as pd

from rag_papers import store
from rag_papers.ingest import DEFAULT_PATH

LOCAL_DIGESTS = Path("digests")


def fetch() -> tuple[Path, Path]:
    """Return (papers.parquet path, digests directory), downloading them if needed."""
    repo = os.environ.get("HF_DATASET_REPO")
    if not repo:
        return DEFAULT_PATH, LOCAL_DIGESTS
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(repo, repo_type="dataset", allow_patterns=["papers.parquet", "digests/*.json"])
    )
    return root / "papers.parquet", root / "digests"


def load_papers(path: Path) -> pd.DataFrame:
    df = store.load(path)
    # Papers the daily job has not embedded yet cannot be indexed.
    return df[df["embedding"].map(lambda e: hasattr(e, "__len__") and len(e) > 0)].reset_index(drop=True)


def latest_digest(directory: Path) -> dict | None:
    files = sorted(directory.glob("*.json"))
    return json.loads(files[-1].read_text()) if files else None
