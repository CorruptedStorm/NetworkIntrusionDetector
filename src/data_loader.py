"""
data_loader.py

Handles downloading and loading the NSL-KDD dataset.
"""

import os
import urllib.request
import pandas as pd

TRAIN_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]


def download_data(data_dir: str = "data") -> None:
    """
    Download the NSL-KDD train/test files into data_dir if not already present.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "KDDTrain.txt")
    test_path = os.path.join(data_dir, "KDDTest.txt")

    if not os.path.exists(train_path):
        urllib.request.urlretrieve(TRAIN_URL, train_path)
    if not os.path.exists(test_path):
        urllib.request.urlretrieve(TEST_URL, test_path)


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the train/test NSL-KDD data into DataFrames with column names and a binary label.
    """
    train_path = os.path.join(data_dir, "KDDTrain.txt")
    test_path = os.path.join(data_dir, "KDDTest.txt")

    train = pd.read_csv(train_path, names=COLUMNS)
    test = pd.read_csv(test_path, names=COLUMNS)

    train["is_attack"] = (train["label"] != "normal").astype(int)
    test["is_attack"] = (test["label"] != "normal").astype(int)

    return train, test
