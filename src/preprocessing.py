"""
preprocessing.py

Encodes categorical features and scales numeric features for model training.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
DROP_COLS = ["label", "difficulty"]


def encode_categorical(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-encode categorical columns, fitting on the combined train+test values
    so both sets share the same encoding."""
    train = train.copy()
    test = test.copy()

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    return train, test


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    """Split into X/y, encode categoricals, and scale numeric features.

    Returns: X_train_scaled, X_test_scaled, y_train, y_test, fitted_scaler
    """
    train = train.drop(columns=DROP_COLS)
    test = test.drop(columns=DROP_COLS)

    train, test = encode_categorical(train, test)

    X_train = train.drop(columns=["is_attack"])
    y_train = train["is_attack"]
    X_test = test.drop(columns=["is_attack"])
    y_test = test["is_attack"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
