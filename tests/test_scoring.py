"""Tests for scoring.py: the local copy of the automarker's matching and F1 scores."""
import numpy as np
import pytest

import scoring


# -------------------------------------------------------------- match_pairs
def test_match_within_tolerance():
    assert scoring.match_pairs([140], [100], tol=50) == [(0, 0)]


def test_no_match_outside_tolerance():
    assert scoring.match_pairs([151], [100], tol=50) == []


def test_matching_is_one_to_one():
    """Two detections of one spike: one hit, the other is a false positive."""
    pairs = scoring.match_pairs([95, 105], [100], tol=50)
    assert len(pairs) == 1


def test_closest_prediction_is_chosen():
    pairs = scoring.match_pairs([130, 102], [100], tol=50)
    assert pairs == [(1, 0)]                          # index of 102 in the input


def test_indices_refer_to_unsorted_inputs():
    pred = np.array([900, 100, 500])
    true = np.array([505, 905, 95])
    for i, j in scoring.match_pairs(pred, true, tol=50):
        assert abs(pred[i] - true[j]) <= 50
    assert len(scoring.match_pairs(pred, true, tol=50)) == 3


def test_empty_inputs():
    assert scoring.match_pairs([], [100]) == []
    assert scoring.match_pairs([100], []) == []


# ----------------------------------------------------------------- f1_score
def test_f1_score_values():
    assert scoring.f1_score(10, 10, 10) == pytest.approx(1.0)
    assert scoring.f1_score(0, 10, 10) == 0.0
    # precision 1.0, recall 0.5 -> F1 = 2/3
    assert scoring.f1_score(5, 5, 10) == pytest.approx(2 / 3)


def test_f1_score_handles_no_predictions():
    assert scoring.f1_score(0, 0, 10) == 0.0


# ------------------------------------------------------------ dataset_score
def test_perfect_prediction_scores_one():
    idx, cls = np.array([100, 500, 900]), np.array([1, 2, 3])
    assert scoring.dataset_score(idx, cls, idx, cls) == pytest.approx((1.0, 1.0, 1.0))


def test_right_place_wrong_class_gives_only_detection_credit():
    idx = np.array([100, 500, 900])
    f_ident, f_class, f_dataset = scoring.dataset_score(idx, np.array([2, 3, 1]),
                                                        idx, np.array([1, 2, 3]))
    assert f_ident == pytest.approx(1.0)
    assert f_class == pytest.approx(0.0)
    assert f_dataset == pytest.approx(0.3)            # 0.3 * 1 + 0.7 * 0


def test_dataset_score_with_misses_and_false_positives():
    true_idx, true_cls = np.array([100, 500, 900, 1300]), np.array([1, 1, 2, 2])
    pred_idx, pred_cls = np.array([105, 510, 2000]), np.array([1, 2, 1])
    # 2 detections match (P = 2/3, R = 2/4); 1 of those has the right class (P = 1/3, R = 1/4)
    f_ident, f_class, f_dataset = scoring.dataset_score(pred_idx, pred_cls, true_idx, true_cls)
    assert f_ident == pytest.approx(2 * (2/3) * (1/2) / (2/3 + 1/2))
    assert f_class == pytest.approx(2 * (1/3) * (1/4) / (1/3 + 1/4))
    assert f_dataset == pytest.approx(0.3 * f_ident + 0.7 * f_class)


# --------------------------------------------------------- recall_per_class
def test_recall_per_class():
    true_idx = np.array([100, 500, 900, 1300])
    true_cls = np.array([1, 1, 3, 3])
    recall = scoring.recall_per_class([100, 900], true_idx, true_cls, classes=[1, 2, 3])
    assert recall[1] == pytest.approx(0.5)
    assert recall[3] == pytest.approx(0.5)
    assert np.isnan(recall[2])                        # no Class 2 spikes at all
