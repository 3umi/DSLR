import pandas as pd

from pathlib import Path

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if 'Index' in df.columns:
        df = df.set_index('Index')
    return df

def numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include='number').dropna(axis=1, how="all")
    if num.empty:
        raise ValueError('no numeric columns in dataset')
    return num

def clean_data(s: pd.Series) -> list[float]:
    return s.dropna().sort_values().to_list()