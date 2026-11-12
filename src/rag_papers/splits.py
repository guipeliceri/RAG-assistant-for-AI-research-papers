"""Dev/test splits that cannot be tuned on by accident.

Three guards:

1. **Assignment by hash.** Whether an item is dev or test depends only on its ID and a salt
   fixed when the label set is created, never on its label, its score or the order it was
   labelled in. Adding more labels later never moves an old item to the other side.
2. **Lock.** When the test labels are final, their fingerprint is written to a lock file.
   Any later change to a test label (fixing "a slip" after seeing the model's mistakes is the
   classic way to inflate a metric) makes evaluation fail until the change is made openly
   with a new lock and a note.
3. **Ledger.** Every evaluation on the test split is appended to a log with the system
   version, so the report says how many times the test set was looked at.
"""

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

LOCK_DIR = Path("eval/locks")
LEDGER = Path("eval/test_runs.jsonl")


class LockError(RuntimeError):
    pass


def split_of(item_id: str, salt: str, test_share: float = 0.5) -> str:
    """'dev' or 'test', from a hash of the salt and the ID only."""
    if not 0 < test_share < 1:
        raise ValueError("test_share must be between 0 and 1")
    digest = hashlib.sha256(f"{salt}:{item_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    return "test" if u < test_share else "dev"


def partition(labels: Mapping[str, object], salt: str, test_share: float = 0.5) -> tuple[dict, dict]:
    dev, test = {}, {}
    for item_id, label in labels.items():
        (test if split_of(item_id, salt, test_share) == "test" else dev)[item_id] = label
    return dev, test


def fingerprint(labels: Mapping[str, object]) -> str:
    canonical = json.dumps(sorted((str(k), v) for k, v in labels.items()), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def lock(name: str, test_labels: Mapping[str, object], note: str, directory: Path = LOCK_DIR) -> Path:
    """Freeze the test labels. Refuses to overwrite: a relock must be a deliberate, visible change."""
    path = directory / f"{name}.json"
    if path.exists():
        raise LockError(f"{path} already exists; delete it in a commit that explains why before relocking")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": name,
                "n": len(test_labels),
                "fingerprint": fingerprint(test_labels),
                "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "note": note,
            },
            indent=2,
        )
        + "\n"
    )
    return path


def verify(name: str, test_labels: Mapping[str, object], directory: Path = LOCK_DIR) -> None:
    path = directory / f"{name}.json"
    if not path.exists():
        raise LockError(f"test set '{name}' is not locked yet; lock it before reporting test metrics")
    locked = json.loads(path.read_text())
    if locked["fingerprint"] != fingerprint(test_labels):
        raise LockError(f"test labels of '{name}' changed after they were locked")


def record_test_run(name: str, version: str, metrics: dict, ledger: Path = LEDGER) -> int:
    """Append one test evaluation to the ledger and return how many runs this test set has had."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "test_set": name,
        "version": version,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    with ledger.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return test_runs(name, ledger)


def test_runs(name: str, ledger: Path = LEDGER) -> int:
    if not ledger.exists():
        return 0
    return sum(json.loads(line)["test_set"] == name for line in ledger.read_text().splitlines() if line.strip())
