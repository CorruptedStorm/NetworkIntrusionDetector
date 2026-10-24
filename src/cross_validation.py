from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, make_scorer
import numpy as np
import pandas as pd


def run_cv(models, X, y, cv, groups=None, scoring=("accuracy", "precision", "recall")):
    means = {}   # [mean accuracy, mean precision, mean recall]
    stds = {}    # [std accuracy, std precision, std recall]
    confusion_matrices = {}  # stores confusion matrix for each model

    scorers = {metric: metric for metric in scoring}   # accuracy/precision/recall, by name
    scorers["tn"] = _confusion_cell(0, 0)
    scorers["fp"] = _confusion_cell(0, 1)
    scorers["fn"] = _confusion_cell(1, 0)
    scorers["tp"] = _confusion_cell(1, 1)

    for name, estimator in models.items():
        results = cross_validate(estimator, X, y, cv=cv, groups=groups, scoring=scorers)

        mean_scores = []
        std_scores = []
        for metric in scoring:
            fold_scores = results[f"test_{metric}"]   # one score per fold
            mean_scores.append(fold_scores.mean())
            std_scores.append(fold_scores.std())

        means[name] = mean_scores
        stds[name] = std_scores


        tn = results["test_tn"].sum()    
        fp = results["test_fp"].sum()
        fn = results["test_fn"].sum()
        tp = results["test_tp"].sum()
        confusion_matrices[name] = np.array([[tn, fp], [fn, tp]]).astype(int)

        print(name + " is done!")





    row_labels = [m.capitalize() for m in scoring]
    return pd.DataFrame(means, index=row_labels), pd.DataFrame(stds, index=row_labels), confusion_matrices


def _confusion_cell(row, col):
    """Returns a scorer that reports ONE cell of the 2x2 confusion matrix."""
    def cell(y_true, y_pred):
        return confusion_matrix(y_true, y_pred, labels=[0, 1])[row, col]
    
    return make_scorer(cell)