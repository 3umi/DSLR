import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# batch_size None = full batch GD, 1 = stochastic GD, k = mini-batch GD
OPTIMIZERS = {
    'batch':     {'lr': 0.1,  'epochs': 1000, 'batch_size': None},
    'minibatch': {'lr': 0.1,  'epochs': 50,   'batch_size': 32},
    'sgd':       {'lr': 0.01, 'epochs': 10,   'batch_size': 1},
}

def step(X: np.ndarray, y: np.ndarray, theta0: float, theta: np.ndarray,
         lr: float) -> tuple[float, np.ndarray]:
    m = len(y)
    err = sigmoid(X @ theta + theta0) - y
    grad0 = np.sum(err) / m
    grad = (X.T @ err) / m
    return theta0 - lr * grad0, theta - lr * grad

def train_one(X: np.ndarray, y: np.ndarray, lr: float, epochs: int,
              batch_size: int | None = None,
              seed: int = 42) -> tuple[float, np.ndarray]:
    m, n = X.shape
    theta0 = 0.0
    theta = np.zeros(n)
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        if batch_size is None:
            theta0, theta = step(X, y, theta0, theta, lr)
            continue
        order = rng.permutation(m)
        for start in range(0, m, batch_size):
            batch = order[start:start + batch_size]
            theta0, theta = step(X[batch], y[batch], theta0, theta, lr)
    return theta0, theta

def predict(X: np.ndarray, houses: list[str],
            weights: dict[str, tuple[float, np.ndarray]]) -> np.ndarray:
    ordered = [weights[house] for house in houses]
    H = np.column_stack([sigmoid(X @ t + t0) for t0, t in ordered])
    return np.array(houses)[np.argmax(H, axis=1)]
