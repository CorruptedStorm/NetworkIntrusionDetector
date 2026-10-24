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