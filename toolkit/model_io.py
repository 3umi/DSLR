import json
from pathlib import Path

import numpy as np

def save_model(path: Path,
               features: list[str],
               means: dict[str, float],
               stds: dict[str, float],
               houses: list[str],
               weights: dict[str, tuple[float, np.ndarray]]) -> None:
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