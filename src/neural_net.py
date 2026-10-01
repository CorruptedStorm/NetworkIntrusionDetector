"""
neural_net.py

A small neural network (one hidden layer) for the same task as model.py,
using scikit-learn's MLPClassifier. Same interface as train_model/predict
in model.py, so it drops into the existing structure.
"""

from sklearn.neural_network import MLPClassifier


def train_mlp(X_train, y_train, hidden_layer_size=32, random_state=42):
    """Train a neural net with ONE hidden layer of hidden_layer_size neurons."""
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),  # one hidden layer
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict_mlp(model, X):
    return model.predict(X)