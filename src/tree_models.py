import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

def train_hgb(X_train, y_train, random_state=42):
    return HistGradientBoostingClassifier(
        class_weight="balanced", random_state=random_state).fit(X_train, y_train)

class BenignOnlyIsolationForest:
    """Fits on benign rows only; predicts 1 (attack) for anything that looks unusual."""
    def __init__(self, **kwargs):
        self.model = IsolationForest(random_state=42, **kwargs)
    def fit(self, X, y):
        self.model.fit(X[np.asarray(y) == 0]); return self
    def predict(self, X):
        return (self.model.predict(X) == -1).astype(int)