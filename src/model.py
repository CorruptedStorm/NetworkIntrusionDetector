"""
model.py

Trains and applies the logistic regression classifier.
"""

from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train) -> LogisticRegression:
    """Train a logistic regression classifier on the scaled training features."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def predict(model: LogisticRegression, X):
    """Return binary predictions (0 = normal, 1 = attack) for the given features."""
    return model.predict(X)
