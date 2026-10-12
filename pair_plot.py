import seaborn as sns
import matplotlib.pyplot as plt

import sys

from pathlib import Path

from toolkit.load import load_csv
from toolkit.groups import group_by_num

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python pair_plot.py <dataset.csv>')
        sys.exit(1)

    path = Path(sys.argv[1])

    try:
        df = load_csv(path)
        courses = list(group_by_num(df).columns)
        data = df[courses + ['Hogwarts House']]
        sns.pairplot(data, hue='Hogwarts House', diag_kind='hist', height=1.2, plot_kws={'s': 5})
        plt.show()
    except (ValueError, OSError) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\naborted.', file=sys.stderr)
        sys.exit(130)