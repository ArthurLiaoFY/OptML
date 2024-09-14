import numpy as np


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return np.mean((y - y_hat) ** 2).item()


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return np.sqrt(np.mean((y - y_hat) ** 2)).item()


def R_square(y: np.ndarray, y_hat: np.ndarray) -> float:
    return (1.0 - (np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))).item()
