"""Model training helpers for the baseline and XGBoost models.

Both helpers return a fitted estimator. Evaluation lives in ``evaluation.py``
so these functions stay narrow and composable.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import RANDOM_SEED


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str | dict | None = "balanced",
) -> LogisticRegression:
    """Fit a logistic regression baseline.

    ``class_weight="balanced"`` adjusts for the ~27% churn rate so the model
    isn't biased toward predicting "stay". We use ``liblinear`` because it's
    stable for small-to-medium tabular data and supports L1/L2.
    """
    model = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        class_weight=class_weight,
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float | None = None,
):
    """Fit an XGBoost classifier with sensible defaults for ~7k rows.

    If ``scale_pos_weight`` is None, it is auto-computed as
    ``n_negative / n_positive`` — the standard XGBoost way to handle
    imbalance without oversampling.
    """
    from xgboost import XGBClassifier  # Lazy import so src/ doesn't hard-require it

    if scale_pos_weight is None:
        pos = int(y_train.sum())
        neg = len(y_train) - pos
        scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
