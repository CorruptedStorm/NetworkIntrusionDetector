import numpy as np
from src.model import train_model, predict


def test_model_trains_and_predicts_correct_shape():
    # Simple separable fake data: two clear clusters
    X_train = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)
    preds = predict(model, X_train)

    assert len(preds) == len(y_train)
    assert set(preds).issubset({0, 1})


def test_model_learns_obvious_pattern():
    # Data is trivially separable, so the model should get this perfectly right
    X_train = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)
    preds = predict(model, X_train)

    assert list(preds) == list(y_train)
