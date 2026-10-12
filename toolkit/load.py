import pandas as pd

from pathlib import Path

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if 'Index' in df.columns:
        df = df.set_index('Index')
    return df

def clean_data(s: pd.Series) -> list[float]:
    return s.dropna().to_list()

def clean_pair(df: pd.DataFrame, col1: str,
               col2: str) -> tuple[list[float], list[float]]:
    feat = df[[col1, col2]].dropna()
    return feat[col1].to_list(), feat[col2].to_list()
