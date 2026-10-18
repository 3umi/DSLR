import json
from pathlib import Path

import numpy as np

Weights = dict[str, tuple[float, np.ndarray]]

def save_model(path: Path, features: list[str], means: dict[str, float],
               stds: dict[str, float], houses: list[str], weights: Weights) -> None:
    model = {
        'features': features,
        'means': means,
        'stds': stds,
        'houses': houses,
        'weights': {house: {'theta0': float(t0), 'theta': t.tolist()}
                    for house, (t0, t) in weights.items()},
    }
    with open(path, 'w') as f:
        json.dump(model, f, indent=2)

def load_model(path: Path) -> tuple[list[str], dict[str, float], dict[str, float],
                                    list[str], Weights]:
    if not path.exists():
        raise OSError('No trained model found. Run "logreg_train.py" first.')
    
    with open(path) as f:
        try:
            data = json.load(f)
        except ValueError as e:
            raise ValueError(f'{path}: not valid JSON ({e})')
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a JSON object')
    
    try:
        features = data['features']
        means = data['means']
        stds = data['stds']
        houses = data['houses']
        model = data['weights']
        weights = {}
        for house, thetas in model.items():
            theta0, theta = float(thetas['theta0']), np.array(thetas['theta'], dtype=float)
            if len(theta) != len(features):
                raise ValueError(f'{house} has {len(theta)} weights, expected {len(features)}')
            weights[house] = theta0, theta
        if sorted(weights) != sorted(houses):
            raise ValueError(f'weights {sorted(weights)} do not match '
                             f'houses {sorted(houses)}')

    except KeyError as e:
        raise ValueError(f'{path}: missing key {e}')
    except (TypeError, ValueError) as e:
        raise ValueError(f'{path}: invalid model ({e})')
    
    return features, means, stds, houses, weights