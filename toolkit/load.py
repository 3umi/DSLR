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
