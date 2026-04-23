"""Evaluation helpers: metrics table, ROC curve, confusion matrix, importances.

These return plain values or matplotlib figures/axes so the notebooks can
compose layouts however they like.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Return the headline classification metrics as a dict."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_roc(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    label: str = "Model",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot an ROC curve on ``ax`` (or a new figure). Returns the axes so the
    caller can overlay multiple models.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return ax


def plot_confusion(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a labeled confusion matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return ax


def feature_importance_df(model, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame of feature importances, sorted descending.

    Works for both ``LogisticRegression`` (``coef_``) and tree models
    (``feature_importances_``). For logistic regression, the value is the
    signed coefficient; use ``.abs()`` to rank by magnitude.
    """
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        label = "importance"
    elif hasattr(model, "coef_"):
        values = model.coef_.ravel()
        label = "coefficient"
    else:
        raise ValueError(f"Model {type(model).__name__} has no importances or coefficients")

    return (
        pd.DataFrame({"feature": feature_names, label: values})
        .assign(abs_value=lambda d: d[label].abs())
        .sort_values("abs_value", ascending=False)
        .drop(columns="abs_value")
        .reset_index(drop=True)
    )
