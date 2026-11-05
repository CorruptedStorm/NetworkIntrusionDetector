"""
compare.py

Generic comparison of any number of trained models against the same test set.
"""

import pandas as pd
from src.evaluate import evaluate

import matplotlib.pyplot as plt
import seaborn as sns


def compare_models(models: dict, X_test, y_test, metrics=("accuracy", "precision", "recall")) -> pd.DataFrame:
    """
    models: dictionary mapping a display name -> a trained model with a .predict() method
    Returns a DataFrame: rows = metrics, columns = model names.
    """
    results = {}
    confusion_matrices = {}

    for name, model in models.items():
        preds = model.predict(X_test)
        model_metrics = evaluate(y_test, preds)
        results[name] = [model_metrics[m] for m in metrics]
        confusion_matrices[name] = model_metrics["confusion_matrix"]

    results_df = pd.DataFrame(results, index=[m.capitalize() for m in metrics])
    return results_df, confusion_matrices


def plot_bar_chart(results_df, title="Model Comparison", path="comparison_bar_chart.png", errors=None):
    results_df.plot(kind="bar", yerr=errors, capsize=4, figsize=(8, 5))
    plt.title(title)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)          # 1.05 so error-bar caps near 1.0 aren't clipped
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path)

def plot_confusion_matrices(confusion_matrices, path="confusion_matrices.png", normalize=False):
    num_models = len(confusion_matrices)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        if normalize:
            cm = cm / cm.sum(axis=1, keepdims=True)   # each row becomes percentages
            fmt = ".2f"
        else:
            fmt = "d"
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=["normal", "attack"], yticklabels=["normal", "attack"], ax=ax)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path)

