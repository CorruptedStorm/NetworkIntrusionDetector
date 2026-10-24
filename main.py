"""
main.py

Runs the full program: load data, preprocess, train, evaluate.

Usage:
    python main.py
"""

from src.kdd_data_loader import kdd_load_data
from src.preprocessing import prepare_kdd_features
from src.model import train_model, predict
from src.evaluate import print_report


def main():

    print("Loading KDD data...")
    train, test = kdd_load_data(data_dir="data/data_kdd")
    print(f"  Training rows: {len(train)}")
    print(f"  Test rows:     {len(test)}")

    print("Preparing features...")
    X_train, X_test, y_train, y_test, _scaler = prepare_kdd_features(train, test)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating...")
    y_pred = predict(model, X_test)
    print_report(y_test, y_pred)


if __name__ == "__main__":
    main()
