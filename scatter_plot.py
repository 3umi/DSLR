import sys
import math

from pathlib import Path

from toolkit.stats import my_correlation
from toolkit.load import load_csv, clean_pair
from toolkit.groups import group_by_num
from toolkit.plot import plot

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python scatter_plot.py <dataset.csv>')
        sys.exit(1)

    path = Path(sys.argv[1])

    try:
        df = load_csv(path)
        courses = list(group_by_num(df).columns)
        best = None
        for i in range(len(courses)):
            for j in range(i + 1, len(courses)):
                xs, ys = clean_pair(df, courses[i], courses[j])
                r = my_correlation(xs, ys)
                if math.isnan(r):
                    continue
                if best is None or abs(r) > abs(best[0]):
                    best = r, courses[i], courses[j]
        if best is None:
            raise ValueError('no pair of features can be correlated')
        r, a, b = best
        xs, ys = clean_pair(df, a, b)
        print(f'Most similar: {a} vs {b} (r = {r:.4f})')
        plot(xs, ys, a, b, r)
    except (ValueError, OSError) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\naborted.', file=sys.stderr)
        sys.exit(130)
