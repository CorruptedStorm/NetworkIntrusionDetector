import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.base import BaseEstimator, ClassifierMixin

def train_hgb(X_train, y_train, learning_rate=0.1, max_iter=100, max_depth=None,
              max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0,
              class_weight="balanced", random_state=42):
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization, class_weight=class_weight,
        random_state=random_state)
    return model.fit(X_train, y_train)

class BenignOnlyIsolationForest(BaseEstimator, ClassifierMixin):
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        self.contamination = contamination      # only store settings here, nothing else
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        benign_rows = X[np.asarray(y) == 0]
        self.model_ = IsolationForest(contamination=self.contamination,
                                      n_estimators=self.n_estimators,
                                      random_state=self.random_state).fit(benign_rows)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        raw = self.model_.predict(X)       # +1 = looks normal, -1 = looks unusual
        return (raw == -1).astype(int)