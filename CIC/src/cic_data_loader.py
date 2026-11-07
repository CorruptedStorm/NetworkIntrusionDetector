import glob, os
import numpy as np
import pandas as pd


def cic_load_data(data_dir="data/data_cic"):
    """Load CIC network intrusion data from CSV files in the specified directory.

    Args:
        data_dir (str, optional): Directory containing the CSV files. Defaults to "data/data_cic".

    Returns:
        pd.DataFrame: DataFrame containing the concatenated and preprocessed data from all CSV files.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))


    if not paths:
        raise FileNotFoundError(f"No CSVs in {data_dir}")
    frames = []


    for path in paths:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        df.columns = df.columns.str.strip()  
        df["Day"] = os.path.basename(path).split("-")[0].lower()
        frames.append(df)


    df = pd.concat(frames, ignore_index=True)
    df = df.drop(columns=["Fwd Header Length.1"], errors="ignore")  # duplicated column
    df = df.replace([np.inf, -np.inf], np.nan).dropna() # this shouldn't be an issue but put it here anyway
    df = df.drop_duplicates()
    df["is_attack"] = (df["Label"].str.strip().str.upper() != "BENIGN").astype(int)



    return df


def split_by_day(df, test_days=("friday",)):
    """Split the DataFrame into training and testing sets based on the specified test days.

    Args:
        df (pd.DataFrame): DataFrame containing the network intrusion data.
        test_days (tuple, optional): Days to be used as the test set. Defaults to ("friday",).

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """
    mask = df["Day"].isin(test_days)
    
    return df[~mask].copy(), df[mask].copy()


def split_within_days(df, test_frac=0.25, n_blocks=20, gap=50, min_block=200,
                      random_state=42):
    """Split every (Day, Label) group into contiguous time blocks and send
    whole blocks to train or test.

    Why blocks (and not random rows): neighbouring flows in an attack run are
    near-duplicates, so a random row split leaks. Whole blocks keep that from
    happening. Doing it per (Day, Label) guarantees every attack family
    appears in BOTH train and test.

    Row order is used as a stand-in for time (the CIC CSVs are roughly
    chronological and your loader keeps that order).

    Args:
        test_frac:    approximate fraction of each group's blocks for test.
        n_blocks:     blocks per (Day, Label) group (fewer for small groups).
        gap:          rows trimmed from each end of every test block and
                      dropped, so train and test never touch directly.
        min_block:    minimum rows per block; small groups get fewer blocks.

    Returns:
        train, test DataFrames. Both have a "Block" column (use it as the
        groups= argument for GroupKFold, and leave it out of the features).
    """
    rng = np.random.default_rng(random_state)
    train_parts, test_parts = [], []

    for (day, label), g in df.groupby(["Day", "Label"], sort=False):
        g = g.sort_index()                      # original (chronological) order
        n = len(g)
        if n < 2:                               # can't split one row
            train_parts.append(g.assign(Block=f"{day}|{label}|0"))
            continue

        k = int(max(2, min(n_blocks, n // min_block)))
        n_test = int(min(k - 1, max(1, round(k * test_frac))))
        test_ids = set(rng.choice(k, size=n_test, replace=False).tolist())

        for b, pos in enumerate(np.array_split(np.arange(n), k)):
            block = g.iloc[pos].assign(Block=f"{day}|{label}|{b}")
            if b in test_ids:
                trim = gap if len(block) > 4 * gap else 0
                test_parts.append(block.iloc[trim:len(block) - trim])
            else:
                train_parts.append(block)

    train = pd.concat(train_parts).sort_index()
    test = pd.concat(test_parts).sort_index()
    return train, test