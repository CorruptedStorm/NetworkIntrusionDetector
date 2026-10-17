import glob, os
import numpy as np
import pandas as pd


def cic_load_data(data_dir="data/data_cic"):
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
    mask = df["Day"].isin(test_days)
    
    return df[~mask].copy(), df[mask].copy()