import pandas as pd
from src.preprocessing import encode_categorical, prepare_features


def make_sample_df(labels):
    """Build a tiny fake dataset with the same columns the real pipeline expects,
    small enough to reason about by hand."""
    n = len(labels)
    return pd.DataFrame({
        "duration": [0] * n,
        "protocol_type": ["tcp", "udp"] * (n // 2) + ["tcp"] * (n % 2),
        "service": ["http"] * n,
        "flag": ["SF"] * n,
        "src_bytes": list(range(n)),
        "dst_bytes": list(range(n)),
        "land": [0] * n,
        "wrong_fragment": [0] * n,
        "urgent": [0] * n,
        "hot": [0] * n,
        "num_failed_logins": [0] * n,
        "logged_in": [0] * n,
        "num_compromised": [0] * n,
        "root_shell": [0] * n,
        "su_attempted": [0] * n,
        "num_root": [0] * n,
        "num_file_creations": [0] * n,
        "num_shells": [0] * n,
        "num_access_files": [0] * n,
        "num_outbound_cmds": [0] * n,
        "is_host_login": [0] * n,
        "is_guest_login": [0] * n,
        "count": [0] * n,
        "srv_count": [0] * n,
        "serror_rate": [0.0] * n,
        "srv_serror_rate": [0.0] * n,
        "rerror_rate": [0.0] * n,
        "srv_rerror_rate": [0.0] * n,
        "same_srv_rate": [0.0] * n,
        "diff_srv_rate": [0.0] * n,
        "srv_diff_host_rate": [0.0] * n,
        "dst_host_count": [0] * n,
        "dst_host_srv_count": [0] * n,
        "dst_host_same_srv_rate": [0.0] * n,
        "dst_host_diff_srv_rate": [0.0] * n,
        "dst_host_same_src_port_rate": [0.0] * n,
        "dst_host_srv_diff_host_rate": [0.0] * n,
        "dst_host_serror_rate": [0.0] * n,
        "dst_host_srv_serror_rate": [0.0] * n,
        "dst_host_rerror_rate": [0.0] * n,
        "dst_host_srv_rerror_rate": [0.0] * n,
        "label": labels,
        "difficulty": [1] * n,
        "is_attack": [0 if l == "normal" else 1 for l in labels],
    })


def test_encode_categorical_returns_numeric_columns():
    train = make_sample_df(["normal", "neptune"])
    test = make_sample_df(["normal", "normal"])

    train_enc, test_enc = encode_categorical(train, test)

    assert pd.api.types.is_numeric_dtype(train_enc["protocol_type"])
    assert pd.api.types.is_numeric_dtype(test_enc["protocol_type"])


def test_prepare_features_shapes_match():
    train = make_sample_df(["normal", "neptune", "normal", "neptune"])
    test = make_sample_df(["normal", "neptune"])

    X_train, X_test, y_train, y_test, scaler = prepare_features(train, test)

    # Same number of rows in, same number of rows out
    assert X_train.shape[0] == len(train)
    assert X_test.shape[0] == len(test)
    # Same number of feature columns between train and test
    assert X_train.shape[1] == X_test.shape[1]
    # Labels preserved correctly
    assert list(y_train) == [0, 1, 0, 1]


def test_prepare_features_scales_to_roughly_zero_mean():
    train = make_sample_df(["normal", "neptune", "normal", "neptune"])
    test = make_sample_df(["normal", "neptune"])

    X_train, _, _, _, _ = prepare_features(train, test)

    # StandardScaler should center training data near 0 mean per column
    assert abs(X_train.mean()) < 1e-6
