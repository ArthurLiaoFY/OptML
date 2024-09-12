import numpy as np


def f(x1, x2):
    return (-np.cos(np.pi * (x1)) * np.cos(2 * np.pi * (x2))) / (
        1 + np.power(x1, 2) + np.power(x2, 2)
    )
