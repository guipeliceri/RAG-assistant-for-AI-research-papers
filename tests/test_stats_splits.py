import numpy as np
import pytest

from rag_papers import splits, stats


def test_wilson_matches_known_value():
    low, high = stats.wilson(8, 10)
    assert low == pytest.approx(0.490, abs=1e-3)
    assert high == pytest.approx(0.943, abs=1e-3)
    assert stats.wilson(0, 0) is None


def test_binary_report_counts_and_intervals():
    truth = [True] * 6 + [False] * 4
    pred = [True, True, True, True, False, False, True, False, False, False]
    r = stats.binary_report(truth, pred)
    assert r["confusion"] == {"tp": 4, "fp": 1, "fn": 2, "tn": 3}
    assert r["precision"]["value"] == 0.8
    assert r["recall"]["value"] == pytest.approx(0.667, abs=1e-3)
    assert r["base_rate"] == 0.6
    low, high = r["precision"]["ci95"]
    assert low < 0.8 < high


def test_always_yes_shows_precision_equal_to_base_rate():
    truth = [True] * 3 + [False] * 7
    r = stats.binary_report(truth, [True] * 10)
    assert r["recall"]["value"] == 1.0
    assert r["precision"]["value"] == r["base_rate"] == 0.3
    assert r["cohen_kappa"]["value"] == 0.0


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(1)
    t = rng.random(50) < 0.5
    p = rng.random(50) < 0.5
    assert stats.binary_report(t, p)["f1"] == stats.binary_report(t, p)["f1"]


def test_calibration_perfect_and_baseline():
    y = [True, False] * 50
    r = stats.calibration(y, [0.5] * 100)
    assert r["ece"] == 0.0
    assert r["brier"] == r["brier_baseline"] == 0.25
    with pytest.raises(ValueError):
        stats.calibration([True], [1.2])


def test_calibration_counts_probability_one():
    r = stats.calibration([True, True], [1.0, 1.0])
    assert r["table"][-1]["n"] == 2 and r["ece"] == 0.0


def test_split_is_stable_and_label_independent():
    ids = [f"2611.{i:05d}" for i in range(400)]
    a = {i: True for i in ids}
    b = {i: False for i in ids}
    dev_a, test_a = splits.partition(a, "salt")
    dev_b, test_b = splits.partition(b, "salt")
    assert set(test_a) == set(test_b)
    assert 150 < len(test_a) < 250
    # Adding items never moves the old ones.
    _, test_more = splits.partition({**a, "new.1": True}, "salt")
    assert set(test_a) <= set(test_more)


def test_lock_detects_changed_labels(tmp_path):
    labels = {"a": True, "b": False}
    splits.lock("demo", labels, "first lock", tmp_path)
    splits.verify("demo", labels, tmp_path)
    with pytest.raises(splits.LockError):
        splits.verify("demo", {"a": True, "b": True}, tmp_path)
    with pytest.raises(splits.LockError):
        splits.lock("demo", labels, "again", tmp_path)
    with pytest.raises(splits.LockError):
        splits.verify("other", labels, tmp_path)


def test_ledger_counts_runs(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    assert splits.test_runs("demo", ledger) == 0
    splits.record_test_run("demo", "v1", {"f1": 0.8}, ledger)
    assert splits.record_test_run("demo", "v2", {"f1": 0.9}, ledger) == 2
    assert splits.test_runs("other", ledger) == 0
