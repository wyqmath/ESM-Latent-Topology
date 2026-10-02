"""AUROC bootstrap with multiplicity preserved and explicit cluster membership.

Sample clusters uniformly with replacement, carry every observed member of each
cluster into the draw, and preserve cluster multiplicity as row sample weights.
The estimand remains the chain-weighted AUROC, not a mean of per-cluster AUCs.
"""
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score


def component_ids(sample_ids, mapping):
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate prediction sample IDs")
    missing = [sid for sid in sample_ids if sid not in mapping or not mapping[sid]]
    if missing:
        raise ValueError(f"Missing component IDs: {missing}")
    return [mapping[sid] for sid in sample_ids]


def weights_for_draw(groups, selected):
    """Every member of a selected group gets its selection count, including repeats."""
    counts = Counter(selected)
    return np.asarray([counts[g] for g in groups], dtype=float)


def weighted_auroc(labels, scores, weights=None):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or len(labels) == 0:
        raise ValueError("Nonempty aligned one-dimensional labels/scores required")
    if not np.all(np.isfinite(scores)) or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Finite scores and binary labels required")
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != labels.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Finite nonnegative aligned weights required")
    if any(weights[labels == k].sum() == 0 for k in (0, 1)):
        return float("nan")
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def bootstrap_auroc(labels, scores, groups, *, B=2000, seed=2026,
                    confidence=0.95, comparison_scores=None):
    """Return summary and all B draws (degenerate draws recorded as NaN).

    Both methods use identical draw weights when comparison_scores is supplied.
    A group with several chains may cause a variable draw size; no chain is
    discarded or collapsed. No retries for one-class draws.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    groups = list(groups)
    if len(groups) != len(labels) or not groups or any(g is None or g == "" for g in groups):
        raise ValueError("A nonempty explicit group ID for every row is required")
    if not isinstance(B, int) or B <= 0 or not 0 < confidence < 1:
        raise ValueError("Positive B and 0 < confidence < 1 required")
    point = weighted_auroc(labels, scores)
    if not np.isfinite(point):
        raise ValueError("Observed prediction set must contain both classes")
    other = None if comparison_scores is None else np.asarray(comparison_scores, dtype=float)
    other_point = None if other is None else weighted_auroc(labels, other)
    units = list(dict.fromkeys(groups))
    unit_index = {g: i for i, g in enumerate(units)}
    membership = np.asarray([unit_index[g] for g in groups])
    rng = np.random.RandomState(seed)
    draws = np.full(B, np.nan)
    difference = None if other is None else np.full(B, np.nan)
    for b in range(B):
        counts = np.bincount(rng.randint(0, len(units), size=len(units)), minlength=len(units))
        weights = counts[membership]
        draws[b] = weighted_auroc(labels, scores, weights)
        if other is not None and np.isfinite(draws[b]):
            difference[b] = draws[b] - weighted_auroc(labels, other, weights)
    def summary(values, observed):
        valid = values[np.isfinite(values)]
        alpha = (1 - confidence) / 2
        interval = [None, None] if not len(valid) else np.quantile(valid, [alpha, 1-alpha]).tolist()
        return {"point": float(observed), "ci95_low": interval[0], "ci95_high": interval[1],
                "valid_draws": int(len(valid)), "degenerate_discarded": int(B-len(valid)),
                "no_conclusion_flag": bool(len(valid) < B / 2)}
    result = summary(draws, point)
    result.update({"B": B, "seed": seed, "confidence": confidence,
                   "n_rows": len(labels), "n_units": len(units),
                   "estimand": "chain_weighted_auroc",
                   "resampling": "uniform units with replacement; preserve every member and multiplicity"})
    if other is not None:
        result["paired_difference"] = summary(difference, point-other_point)
        result["comparison_auroc"] = other_point
    return result, draws, difference
