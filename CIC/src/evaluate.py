"""
evaluate.py

Computes evaluation metrics for model predictions.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

import pandas as pd
import numpy as np

def evaluate(y_true, y_pred) -> dict:
    """Return a dict of accuracy, precision, recall, and the confusion matrix.

    Args:
        y_true (pd.Series or np.ndarray): True labels.
        y_pred (pd.Series or np.ndarray): Predicted labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and confusion matrix.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def print_report(y_true, y_pred) -> None:
    """Print a full human-readable evaluation report."""
    metrics = evaluate(y_true, y_pred)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print()
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(metrics["confusion_matrix"])
    print()
    print(classification_report(y_true, y_pred, target_names=["normal", "attack"]))


def per_attack_recall(labels, y_pred):
    """Compute the recall rate for each attack type.

    Args:
        labels (pd.Series or np.ndarray): True labels.
        y_pred (pd.Series or np.ndarray): Predicted labels.

    Returns:
        pd.DataFrame: DataFrame containing the recall rate and number of samples for each attack type.
    """
    
    out = (pd.DataFrame({"Label": pd.Series(labels).str.strip().values,
                         "flagged": np.asarray(y_pred) == 1})
             .groupby("Label")["flagged"].agg(rate="mean", rows="size"))
    return out.sort_values("rate")
