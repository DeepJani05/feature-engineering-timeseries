"""Walk-forward cross-validation for time series.

The only honest CV for financial data. Train on a trailing window,
test on the bars *immediately after* it, roll, repeat. Random shuffles
let future information leak into the training set and inflate every
metric you care about — usually by enough to make a losing strategy
look like a winning one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- types


@dataclass
class FoldResult:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    metrics: dict[str, float]


@dataclass
class CVResult:
    folds: list[FoldResult] = field(default_factory=list)
    feature_importance: pd.Series | None = None

    def summary(self) -> pd.DataFrame:
        if not self.folds:
            return pd.DataFrame()
        rows = []
        for f in self.folds:
            row = {
                "fold": f.fold_idx,
                "train_start": f.train_start,
                "test_start": f.test_start,
                "n_train": f.n_train,
                "n_test": f.n_test,
                **f.metrics,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def aggregate_metrics(self) -> dict[str, float]:
        """Mean of each metric across folds."""
        if not self.folds:
            return {}
        keys = set().union(*(f.metrics.keys() for f in self.folds))
        return {
            k: float(np.mean([f.metrics.get(k, np.nan) for f in self.folds]))
            for k in keys
        }


# --------------------------------------------------------- fold iterator


def iter_folds(
    n_samples: int,
    train_window: int,
    test_window: int,
    step: int | None = None,
) -> Iterator[tuple[slice, slice]]:
    """Yield (train_slice, test_slice) pairs over a length-n index.

    The first train window starts at 0; subsequent windows roll forward
    by ``step``. If ``step`` is None, defaults to ``test_window`` (no overlap).
    """
    if step is None:
        step = test_window
    start = 0
    while start + train_window + test_window <= n_samples:
        train = slice(start, start + train_window)
        test = slice(start + train_window, start + train_window + test_window)
        yield train, test
        start += step


# ----------------------------------------------------------- core engine


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    train_model_fn: Callable[[pd.DataFrame, pd.Series], object],
    predict_fn: Callable[[object, pd.DataFrame], np.ndarray],
    metric_fns: dict[str, Callable[[pd.Series, np.ndarray], float]],
    *,
    train_window: int = 504,
    test_window: int = 21,
    step: int | None = None,
) -> CVResult:
    """Run a walk-forward CV over a feature matrix and label series.

    Parameters
    ----------
    X, y : aligned features and labels (same index, no NaNs).
    train_model_fn : (X_train, y_train) -> fitted model
    predict_fn : (model, X_test) -> 1-D numpy array of predictions
    metric_fns : name -> (y_true, y_pred) -> float
    """
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")

    folds: list[FoldResult] = []
    importance_accum = pd.Series(0.0, index=X.columns)
    importance_count = 0

    for fold_idx, (tr, te) in enumerate(iter_folds(len(X), train_window, test_window, step)):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te, y_te = X.iloc[te], y.iloc[te]

        model = train_model_fn(X_tr, y_tr)
        preds = np.asarray(predict_fn(model, X_te))

        fold_metrics = {name: float(fn(y_te, preds)) for name, fn in metric_fns.items()}

        folds.append(
            FoldResult(
                fold_idx=fold_idx,
                train_start=X.index[tr.start],
                train_end=X.index[tr.stop - 1],
                test_start=X.index[te.start],
                test_end=X.index[te.stop - 1],
                n_train=tr.stop - tr.start,
                n_test=te.stop - te.start,
                metrics=fold_metrics,
            )
        )

        if hasattr(model, "feature_importances_"):
            importance_accum += pd.Series(model.feature_importances_, index=X.columns)
            importance_count += 1

        logger.info(
            "walk_forward.fold_complete",
            extra={"fold": fold_idx, **fold_metrics},
        )

    importance = (
        (importance_accum / importance_count).sort_values(ascending=False)
        if importance_count
        else None
    )
    return CVResult(folds=folds, feature_importance=importance)


# -------------------------------------------------------------- metrics


def hit_rate(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Fraction of times sign(y_pred) matches sign(y_true)."""
    if len(y_true) == 0:
        return 0.0
    return float((np.sign(y_pred) == np.sign(y_true)).mean())


def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Same as hit rate but ignores zero-sign predictions."""
    mask = np.sign(y_pred) != 0
    if not mask.any():
        return 0.0
    return float((np.sign(y_pred[mask]) == np.sign(y_true.values[mask])).mean())


def information_coefficient(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predictions and outcomes."""
    if len(y_true) < 2:
        return 0.0
    return float(pd.Series(y_pred, index=y_true.index).corr(y_true, method="spearman"))
