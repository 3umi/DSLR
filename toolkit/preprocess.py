import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype

from toolkit.load import clean_data
from toolkit.stats import my_count, my_mean, my_std

def check_features(df: pd.DataFrame, features: list[str]) -> None:
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f'{feature} is not in the dataset')
        if not is_numeric_dtype(df[feature]):
            raise ValueError(f'{feature} is not numeric')

def fit_scaler(df: pd.DataFrame, features: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means = {}
    stds = {}
    check_features(df, features)
    for feature in features:
        clean = clean_data(df[feature])
        if my_count(clean) < 2:
            raise ValueError(f'{feature} has too few values to scale')
        means[feature] = my_mean(clean)
        stds[feature] = my_std(clean)
        if stds[feature] == 0:
            raise ValueError(f'{feature} has the same value for every student, cannot scale')
    return means, stds

def scale(df: pd.DataFrame, features: list[str], means: dict[str, float], stds: dict[str, float]) -> np.ndarray:
    check_features(df, features)
    for feature in features:
        if feature not in means or feature not in stds:
            raise ValueError(f'{feature} has no scaling parameters')
    raw = df[features].to_numpy(dtype=float)
    mean_array = np.array([means[f] for f in features])
    std_array = np.array([stds[f] for f in features])
    X = (raw - mean_array) / std_array
    X[np.isnan(X)] = 0.0
    return X