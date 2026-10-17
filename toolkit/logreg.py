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