"""
main.py

Runs the full program: download data, preprocess, train, evaluate.

Usage:
    python main.py
"""

from src.data_loader import download_data, load_data
from src.preprocessing import prepare_features
from src.model import train_model, predict
from src.evaluate import print_report


def main():
    print("Downloading data (skips if already present)...")
    download_data(data_dir="data")

    print("Loading data...")
    train, test = load_data(data_dir="data")
    print(f"  Training rows: {len(train)}")
    print(f"  Test rows:     {len(test)}")

    print("Preparing features...")
    X_train, X_test, y_train, y_test, _scaler = prepare_features(train, test)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating...")
    y_pred = predict(model, X_test)
    print_report(y_test, y_pred)


if __name__ == "__main__":
    main()
