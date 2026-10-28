import sys

from pathlib import Path

import pandas as pd

from toolkit.load import load_csv
from toolkit.model_io import load_model
from toolkit.preprocess import scale
from toolkit.logreg import predict

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python logreg_predict.py <test_dataset.csv> <weights.json>')
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    weights_path =  Path(sys.argv[2])

    try:
        df = load_csv(dataset_path)
        features, means, stds, houses, weights = load_model(weights_path)
        x_test = scale(df, features, means, stds)
        predicted = predict(x_test, houses, weights)
        result = pd.DataFrame({
            'Index': df.index,
            'Hogwarts House': predicted,
        })
        result.to_csv('houses.csv', index=False)
        print('Predictions saved to houses.csv')
    except (ValueError, OSError) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\naborted.', file=sys.stderr)
        sys.exit(130)
