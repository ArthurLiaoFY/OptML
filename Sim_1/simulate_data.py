from configparser import ConfigParser

import pandas as pd
from simulate_funcs import f, np
from sklearn.model_selection import train_test_split

config = ConfigParser()
config.read("config.ini")


seed = np.random.RandomState(int(config["simulate"].get("seed")))


xy_min = [-2.0, -2.0]
xy_max = [2.0, 2.0]

model_matrix = seed.uniform(
    low=xy_min,
    high=xy_max,
    size=(int(config["simulate"].get("simulate_size")), len(xy_max)),
)
y = np.array(
    [
        f(x[0], x[1]) + seed.randn() * float(config["simulate"].get("variation_ratio"))
        for x in model_matrix
    ]
)

train_X, val_X, train_y, val_y = train_test_split(
    model_matrix, y, train_size=0.8, random_state=1122
)
