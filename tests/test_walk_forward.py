"""Tests for the walk-forward CV engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.walk_forward import (
    directional_accuracy,
    hit_rate,
    information_coefficient,
    iter_folds,
    walk_forward_cv,
)


def test_iter_folds_basic():
    folds = list(iter_folds(n_samples=100, train_window=50, test_window=10, step=10))
    assert len(folds) == 5
    # First fold: train [0,50), test [50,60)
    first_train, first_test = folds[0]
    assert (first_train.start, first_train.stop) == (0, 50)
    assert (first_test.start, first_test.stop) == (50, 60)
    # Last fold ends exactly at n_samples
    _, last_test = folds[-1]
    assert last_test.stop == 100


def test_iter_folds_no_overlap_by_default():
    folds = list(iter_folds(n_samples=200, train_window=100, test_window=20))
    # step defaults to test_window -> consecutive test sets don't overlap
    test_starts = [t.start for _, t in folds]
    test_ends = [t.stop for _, t in folds]
    assert test_starts == [100, 120, 140, 160, 180]
    assert test_ends == [120, 140, 160, 180, 200]


def test_iter_folds_too_short_yields_nothing():
    assert list(iter_folds(n_samples=10, train_window=50, test_window=10)) == []


def test_walk_forward_cv_runs_end_to_end():
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    X = pd.DataFrame(rng.normal(size=(n, 5)), index=idx, columns=list("abcde"))
    y = pd.Series(rng.normal(size=n), index=idx)

    def _train(X_tr, y_tr):
        from sklearn.linear_model import Ridge

        return Ridge().fit(X_tr, y_tr)

    def _predict(model, X_te):
        return model.predict(X_te)

    metrics = {"hit_rate": hit_rate, "ic": information_coefficient}
    result = walk_forward_cv(
        X, y, _train, _predict, metrics,
        train_window=100, test_window=20,
    )
    assert len(result.folds) >= 5
    summary = result.summary()
    assert "hit_rate" in summary.columns
    assert "ic" in summary.columns
    agg = result.aggregate_metrics()
    assert "hit_rate" in agg


def test_hit_rate_basic():
    y_true = pd.Series([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([0.5, -0.2, 0.1, 0.3])  # last one wrong
    assert hit_rate(y_true, y_pred) == 0.75


def test_directional_accuracy_skips_zero_predictions():
    y_true = pd.Series([1.0, -1.0, 1.0])
    y_pred = np.array([0.5, 0.0, 0.1])  # zero ignored, both nonzero correct
    assert directional_accuracy(y_true, y_pred) == 1.0


def test_information_coefficient_perfect_positive():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0])  # rank-identical
    assert information_coefficient(y_true, y_pred) == pytest.approx(1.0)


def test_mismatched_lengths_raise():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 2])
    with pytest.raises(ValueError, match="length mismatch"):
        walk_forward_cv(X, y, lambda a, b: None, lambda m, x: x, {})
