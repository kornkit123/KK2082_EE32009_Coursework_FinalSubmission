"""Scoring that mirrors the automarker: one-to-one matching and F1 scores."""
import numpy as np
from config import F_CLASS_WEIGHT, F_IDENT_WEIGHT, MATCH_TOLERANCE


def match_pairs(pred, true, tol=MATCH_TOLERANCE):
    """Match predicted spikes to true spikes one-to-one, within +/- `tol` samples.

    Each true spike is matched to the closest unused prediction among its two
    nearest neighbours in time. A prediction can match at most one true spike,
    so two detections of the same spike count as one hit and one false positive.

    Args:
        pred: predicted spike positions (any order).
        true: true spike positions (any order).
        tol:  maximum allowed distance in samples (the brief uses 50).

    Returns:
        List of (pred_i, true_j) pairs, indices into the original `pred` and `true`.
    """
    pred, true = np.asarray(pred), np.asarray(true)
    order = np.argsort(pred)
    pred_sorted = pred[order]
    used = np.zeros(len(pred), bool)
    pairs = []
    for jt in np.argsort(true):
        t = true[jt]
        j = np.searchsorted(pred_sorted, t)
        best = None
        for k in (j - 1, j):          # the predictions just before and just after t
            if 0 <= k < len(pred_sorted) and not used[k] and abs(pred_sorted[k] - t) <= tol:
                if best is None or abs(pred_sorted[k] - t) < abs(pred_sorted[best] - t):
                    best = k
        if best is not None:
            used[best] = True
            pairs.append((order[best], jt))
    return pairs


def f1_score(hits, n_pred, n_true):
    """F1 = harmonic mean of precision (hits / n_pred) and recall (hits / n_true).

    Returns 0 when there are no hits, and never divides by zero.
    """
    precision = hits / max(n_pred, 1)
    recall = hits / max(n_true, 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def dataset_score(pred_idx, pred_cls, true_idx, true_cls, tol=MATCH_TOLERANCE):
    """Score one dataset the same way as the automarker.

    - F_ident: F1 of detection (a hit is any prediction within tol of a true spike).
    - F_class: F1 where a hit also needs the correct class.
    - F_dataset = 0.3 * F_ident + 0.7 * F_class.

    Args:
        pred_idx, pred_cls: predicted positions and classes.
        true_idx, true_cls: true positions and classes (same label convention as pred_cls).
        tol:                matching tolerance in samples.

    Returns:
        (F_ident, F_class, F_dataset)
    """
    pairs = match_pairs(pred_idx, true_idx, tol)
    n_found = len(pairs)
    n_correct = sum(pred_cls[i] == true_cls[j] for i, j in pairs)

    f_ident = f1_score(n_found, len(pred_idx), len(true_idx))
    f_class = f1_score(n_correct, len(pred_idx), len(true_idx))
    return f_ident, f_class, F_IDENT_WEIGHT * f_ident + F_CLASS_WEIGHT * f_class


def recall_per_class(pred_idx, true_idx, true_cls, classes=range(1, 6), tol=MATCH_TOLERANCE):
    """Fraction of true spikes of each class that were detected (class ignored).

    Useful for spotting a class the detector struggles with (e.g. small Class 3 spikes).

    Returns:
        dict {class: recall}; NaN for a class with no true spikes.
    """
    true_cls = np.asarray(true_cls)
    found = np.zeros(len(true_idx), bool)
    for _, j in match_pairs(pred_idx, true_idx, tol):
        found[j] = True
    return {k: found[true_cls == k].mean() if np.any(true_cls == k) else np.nan
            for k in classes}
