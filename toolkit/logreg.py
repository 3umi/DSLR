import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def train_one(X: np.ndarray, y: np.ndarray, lr: float, epochs: int) -> tuple[float, np.ndarray]:
    m, n = X.shape
    theta0 = 0.0
    theta = np.zeros(n)

    for _ in range(epochs):
        z = X @ theta + theta0
        h = sigmoid(z)
        err = h - y
        grad0 = np.sum(err) / m
        grad = (X.T @ err) / m
        theta0 -= lr * grad0
        theta -= lr * grad
    return theta0, theta

def predict(X: np.ndarray, houses: list[str],
            weights: dict[str, tuple[float, np.ndarray]]) -> np.ndarray:
    ordered = [weights[house] for house in houses]
    H = np.column_stack([sigmoid(X @ t + t0) for t0, t in ordered])
    return np.array(houses)[np.argmax(H, axis=1)]
