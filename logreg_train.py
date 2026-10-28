import sys

from pathlib import Path

import numpy as np

from toolkit.load import load_csv
from toolkit.groups import house_labels
from toolkit.preprocess import fit_scaler, scale
from toolkit.logreg import train_one, predict
from toolkit.model_io import save_model


FEATURES = [
    'Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes',
    'History of Magic', 'Transfiguration', 'Charms', 'Flying',
]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python logreg_train.py <dataset.csv>')
        sys.exit(1)

    path = Path(sys.argv[1])

    try:
        df = load_csv(path)
        features = FEATURES
        means, stds = fit_scaler(df, features)
        houses = house_labels(df)
        weights = {}
        X = scale(df, features, means, stds)
        for house in houses:
            y = (df['Hogwarts House'] == house).to_numpy(dtype=float)
            theta0, theta = train_one(X, y, lr=0.1, epochs=1000)
            weights[house] = (theta0, theta)
        predicted = predict(X, houses, weights)
        accuracy = np.mean(predicted == df['Hogwarts House'].to_numpy())
        save_model(Path('weights.json'), features, means, stds, houses, weights)
        print('Model saved to weights.json')
        print(f'Training accuracy: {accuracy * 100:.2f}%')
    except (ValueError, OSError) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\naborted.', file=sys.stderr)
        sys.exit(130)
