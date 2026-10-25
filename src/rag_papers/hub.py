"""Sync the store and digests with the Hugging Face Dataset (the source of truth).

    python -m rag_papers.hub pull
    python -m rag_papers.hub push
"""

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from rag_papers.digest import DIGEST_DIR
from rag_papers.ingest import DEFAULT_PATH

PAPERS_IN_REPO = "papers.parquet"


def repo_id() -> str:
    return os.environ["HF_DATASET_REPO"]


def pull(path: Path = DEFAULT_PATH) -> bool:
    """Download the store. Returns False if the dataset or file does not exist yet."""
    try:
        downloaded = hf_hub_download(repo_id(), PAPERS_IN_REPO, repo_type="dataset")
    except (RepositoryNotFoundError, EntryNotFoundError):
        logging.info("no store on the Hub yet")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(Path(downloaded).read_bytes())
    return True


def push(path: Path = DEFAULT_PATH) -> None:
    api = HfApi()
    api.create_repo(repo_id(), repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=PAPERS_IN_REPO,
        repo_id=repo_id(),
        repo_type="dataset",
        commit_message="Update papers",
    )
    if DIGEST_DIR.exists():
        api.upload_folder(
            folder_path=str(DIGEST_DIR),
            path_in_repo="digests",
            repo_id=repo_id(),
            repo_type="dataset",
            allow_patterns=["*.json"],
            commit_message="Update digests",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pull", "push"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.action == "pull":
        found = pull()
        # Tell the workflow whether this is the first run (backfill needed).
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as out:
                out.write(f"found={'true' if found else 'false'}\n")
    else:
        push()


if __name__ == "__main__":
    main()
