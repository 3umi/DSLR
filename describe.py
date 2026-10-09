import sys

from pathlib import Path

import pandas as pd

from toolkit.load import load_csv, clean_data
from toolkit.groups import group_by_num
from toolkit.stats import (my_count, my_mean, my_std, my_min, my_max,
                           my_percentile)

STATS = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

def describe(df: pd.DataFrame) -> pd.DataFrame:
    data = {}

    df = group_by_num(df)
    for col in df.columns:
        v = clean_data(df[col])
        data[col] = [my_count(v), my_mean(v), my_std(v), my_min(v),
                     my_percentile(v, 25), my_percentile(v, 50),
                     my_percentile(v, 75), my_max(v)]
    return pd.DataFrame(data, index=STATS)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python describe.py <dataset.csv>')
        sys.exit(1)

    path = Path(sys.argv[1])

    try:
        df = load_csv(path)
        print(describe(df).to_string())

    except (ValueError, OSError) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\naborted.', file=sys.stderr)
        sys.exit(130)
