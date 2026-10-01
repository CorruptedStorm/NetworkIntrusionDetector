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
    models: dict mapping a display name -> a trained model with a .predict() method
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


def plot_comparison(results_df, confusion_matrices, bar_chart_path="comparison_bar_chart.png",
                     confusion_path="confusion_matrices.png"):
    # Bar chart: works for any number of models/metrics automatically
    results_df.plot(kind="bar", figsize=(8, 5))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(bar_chart_path)

    # Confusion matrices: one subplot per model, sized dynamically
    num_models = len(confusion_matrices)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]  # keep indexing consistent when there's only one model

    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["normal", "attack"], yticklabels=["normal", "attack"], ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(confusion_path)