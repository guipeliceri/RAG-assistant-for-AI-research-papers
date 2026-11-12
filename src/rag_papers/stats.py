"""Honest metrics: point estimates always come with a 95% interval and the sample size.

Used by every tab's evaluation. Nothing here looks at which split the data came from;
that is the job of `splits.py`.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np

Z95 = 1.959964


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for a proportion. Better than the normal approximation at small n or near 0/1."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _proportion(successes: int, n: int) -> dict:
    interval = wilson(successes, n)
    return {
        "value": round(successes / n, 3) if n else None,
        "ci95": [round(interval[0], 3), round(interval[1], 3)] if interval else None,
        "n": n,
    }


def bootstrap_ci(
    statistic: Callable[..., float],
    *columns: Sequence,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap over items (rows are resampled together). Fixed seed, so reruns match."""
    arrays = [np.asarray(c) for c in columns]
    n = len(arrays[0])
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        v = statistic(*(a[idx] for a in arrays))
        if not math.isnan(v):
            values.append(v)
    if not values:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _f1(truth: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    observed = float(np.mean(a == b))
    pa, pb = float(np.mean(a)), float(np.mean(b))
    expected = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def binary_report(truth: Sequence[bool], predicted: Sequence[bool]) -> dict:
    """Precision, recall, F1, accuracy and Cohen's kappa against human labels, each with a 95% CI.

    The confusion counts are reported too, so a classifier that says "yes" to everything
    cannot hide behind a good recall.
    """
    t = np.asarray(truth, dtype=bool)
    p = np.asarray(predicted, dtype=bool)
    if len(t) != len(p):
        raise ValueError("truth and predicted have different lengths")
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    tn = int(np.sum(~t & ~p))
    f1 = _f1(t, p)
    kappa = _kappa(t, p) if len(t) else float("nan")
    f1_ci = bootstrap_ci(_f1, t, p)
    kappa_ci = bootstrap_ci(_kappa, t, p)
    return {
        "n": len(t),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": _proportion(tp, tp + fp),
        "recall": _proportion(tp, tp + fn),
        "accuracy": _proportion(tp + tn, len(t)),
        "f1": {
            "value": None if math.isnan(f1) else round(f1, 3),
            "ci95": [round(x, 3) for x in f1_ci] if f1_ci else None,
        },
        "cohen_kappa": {
            "value": None if math.isnan(kappa) else round(kappa, 3),
            "ci95": [round(x, 3) for x in kappa_ci] if kappa_ci else None,
        },
        # The share of positives in the labels: what "always yes" would score as precision.
        "base_rate": round(float(np.mean(t)), 3) if len(t) else None,
    }


def brier(outcomes: Sequence[bool], probabilities: Sequence[float]) -> float:
    """Mean squared error of probabilities. 0 is perfect; always answering the base rate is the baseline."""
    y = np.asarray(outcomes, dtype=float)
    q = np.asarray(probabilities, dtype=float)
    return float(np.mean((q - y) ** 2)) if len(y) else float("nan")


def calibration(outcomes: Sequence[bool], probabilities: Sequence[float], bins: int = 10) -> dict:
    """Reliability table and expected calibration error (ECE): does "70% sure" come true 70% of the time?"""
    y = np.asarray(outcomes, dtype=float)
    q = np.asarray(probabilities, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    edges = np.linspace(0, 1, bins + 1)
    # The last bin is closed on the right so a probability of 1.0 is counted.
    which = np.clip(np.digitize(q, edges[1:-1], right=False), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        mask = which == b
        count = int(mask.sum())
        if not count:
            continue
        mean_p, observed = float(q[mask].mean()), float(y[mask].mean())
        ece += count / len(y) * abs(mean_p - observed)
        table.append(
            {
                "bin": [round(edges[b], 2), round(edges[b + 1], 2)],
                "n": count,
                "mean_predicted": round(mean_p, 3),
                "observed": round(observed, 3),
            }
        )
    base = float(y.mean()) if len(y) else float("nan")
    return {
        "n": len(y),
        "brier": round(brier(y, q), 4) if len(y) else None,
        # Brier of always predicting the base rate. A model must beat this to be worth anything.
        "brier_baseline": round(brier(y, np.full(len(y), base)), 4) if len(y) else None,
        "ece": round(ece, 4) if len(y) else None,
        "table": table,
    }


def _weighted_pr(t: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    tp = float(np.sum(w * (t & p)))
    fp = float(np.sum(w * (~t & p)))
    fn = float(np.sum(w * (t & ~p)))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return precision, recall


def weighted_report(
    truth: Sequence[bool],
    predicted: Sequence[bool],
    weights: Sequence[float],
    strata: Sequence[object],
    resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Precision and recall for a stratified sample, estimated for the whole population.

    Label samples over-represent papers similar to the profile, so that enough positives get
    labelled. Without weights, precision and recall would describe the sample, not the stream
    of papers the app shows. Each item's weight is its stratum size over the items sampled from
    that stratum. Intervals come from a bootstrap that resamples within each stratum, the way
    the sample was drawn.
    """
    t = np.asarray(truth, dtype=bool)
    p = np.asarray(predicted, dtype=bool)
    w = np.asarray(weights, dtype=float)
    s = np.asarray(strata)
    if not (len(t) == len(p) == len(w) == len(s)):
        raise ValueError("truth, predicted, weights and strata must have the same length")
    precision, recall = _weighted_pr(t, p, w) if len(t) else (float("nan"), float("nan"))
    groups = [np.flatnonzero(s == g) for g in np.unique(s)] if len(s) else []
    rng = np.random.default_rng(seed)
    boot_p, boot_r = [], []
    for _ in range(resamples if len(t) else 0):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        bp, br = _weighted_pr(t[idx], p[idx], w[idx])
        if not math.isnan(bp):
            boot_p.append(bp)
        if not math.isnan(br):
            boot_r.append(br)

    def summary(value: float, boot: list[float]) -> dict:
        ci = np.percentile(boot, [2.5, 97.5]) if boot else None
        return {
            "value": None if math.isnan(value) else round(value, 3),
            "ci95": [round(float(x), 3) for x in ci] if ci is not None else None,
        }

    return {
        "n": len(t),
        "positives_labelled": int(t.sum()),
        "precision": summary(precision, boot_p),
        "recall": summary(recall, boot_r),
        # Estimated share of relevant papers in the whole stream.
        "base_rate": round(float(np.sum(w * t) / np.sum(w)), 3) if len(t) else None,
    }
