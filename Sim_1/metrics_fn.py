import numpy as np


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return np.mean((y - y_hat) ** 2)


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return np.sqrt(np.mean((y - y_hat) ** 2))


def R_square(y: np.ndarray, y_hat: np.ndarray) -> float:
    return 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))
