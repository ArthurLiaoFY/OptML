import numpy as np

def mse(y, y_hat):
    return np.mean((y - y_hat)**2)

def Rsquare(y, y_hat):
    return 1-(np.sum((y - y_hat)**2)/np.sum((y - y.mean())**2))
