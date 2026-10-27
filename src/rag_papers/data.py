"""Where the app reads papers and digests from.

On Streamlit Community Cloud, `HF_DATASET_REPO` is set (as a root-level secret, which Streamlit
exposes as an environment variable) and everything comes from the Dataset
(the daily job writes there). Locally, the files in the working directory are used.
"""

import json
import os
from pathlib import Path

import pandas as pd

from rag_papers import store
from rag_papers.ingest import DEFAULT_PATH
from rag_papers.popular import COLUMNS as POPULAR_COLUMNS
from rag_papers.popular import DEFAULT_PATH as POPULAR_PATH

LOCAL_DIGESTS = Path("digests")


def fetch() -> tuple[Path, Path, Path]:
    """Return (papers.parquet, digests directory, popular.parquet), downloading them if needed."""
    repo = os.environ.get("HF_DATASET_REPO")
    if not repo:
        return DEFAULT_PATH, LOCAL_DIGESTS, POPULAR_PATH
    from huggingface_hub import snapshot_download

    patterns = ["papers.parquet", "popular.parquet", "digests/*.json"]
    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=patterns))
    return root / "papers.parquet", root / "digests", root / "popular.parquet"


def load_papers(path: Path) -> pd.DataFrame:
    df = store.load(path)
    # Papers the daily job has not embedded yet cannot be indexed.
    return df[df["embedding"].map(lambda e: hasattr(e, "__len__") and len(e) > 0)].reset_index(drop=True)


def latest_digest(directory: Path) -> dict | None:
    files = sorted(directory.glob("*.json"))
    return json.loads(files[-1].read_text()) if files else None


def load_popular(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=POPULAR_COLUMNS)
