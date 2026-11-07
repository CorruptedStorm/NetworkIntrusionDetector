"""
model.py

Trains and applies the logistic regression classifier.
"""

from sklearn.linear_model import LogisticRegression


def train_lr_model(X_train, y_train, c=1.0, max_iter=1000) -> LogisticRegression:
    """Train a logistic regression classifier on the scaled training features."""
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def predict(model: LogisticRegression, X):
    """Return binary predictions (0 = normal, 1 = attack) for the given features."""
    return model.predict(X)
