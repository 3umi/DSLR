import sys
import math

from pathlib import Path

import matplotlib.pyplot as plt

from toolkit.groups import group_by_house, group_by_course, group_by_num
from toolkit.load import load_csv, clean_data
from toolkit.stats import my_min, my_max

BINS = 15
COLS = 5

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python histogram.py <dataset.csv>')
        sys.exit(1)

    path = Path(sys.argv[1])

    try:
        df = load_csv(path)
        courses_names = group_by_num(df).columns
        rows = math.ceil(len(courses_names) / COLS)
        fig, axes = plt.subplots(nrows=rows, ncols=COLS,
                                 figsize=(17, 2.7 * rows))
        axes = axes.flatten()
        groups = group_by_house(df)
        for (i, course) in enumerate(courses_names):
            ax = axes[i]
            courses_dict = group_by_course(groups, course)
            all_scores = clean_data(df[course])
            low = my_min(all_scores)
            width = (my_max(all_scores) - low) / BINS
            edges = [low + k * width for k in range(BINS + 1)]
            for (house, students) in courses_dict.items():
                ax.hist(students, bins=edges, alpha=0.7,
                        label=house, density=True)
            ax.set_title(course)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        for ax in axes[len(courses_names):]:
            ax.axis('off')
        plt.show()

    except (ValueError, OSError) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\naborted.', file=sys.stderr)
        sys.exit(130)
