"""Feature engineering for the Telco Customer Churn dataset.

This module produces a model-ready ``X, y`` split from the cleaned dataframe:
- Drops the customer ID column.
- One-hot encodes all categorical features (drop_first=True for binary cols).
- Standardizes numeric features using ``StandardScaler`` when requested.

Design choice: we return plain numpy arrays / DataFrames rather than a full
sklearn ``Pipeline`` so the notebooks can inspect the transformed matrix
easily. The cost-sensitive layer in Week 2 will wrap this in a Pipeline.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    ID_COL,
    NUMERIC_COLS,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
)


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the cleaned dataframe into ``(X, y)`` with one-hot encoding.

    The target is binarized: ``Churn == "Yes"`` -> 1, else 0.
    """
    df = df.drop(columns=[ID_COL])
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])

    categorical = [c for c in X.columns if c not in NUMERIC_COLS]
    X = pd.get_dummies(X, columns=categorical, drop_first=True)

    # sklearn doesn't like bool columns from get_dummies in some versions
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    return X, y


def train_test_split_scaled(
    X: pd.DataFrame,
    y: pd.Series,
    scale: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with optional scaling of numeric columns.

    Scaling is fit on the training set only to avoid leakage, then applied to
    both splits. Tree-based models (XGBoost) don't need scaling; logistic
    regression does.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    if scale:
        scaler = StandardScaler()
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
        X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return X_train, X_test, y_train, y_test
