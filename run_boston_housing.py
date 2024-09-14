# %%
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from common.optimize_response import optimize_f_hat
from common.plot_fn import plot_obj_surface

config = ConfigParser()
config.read("config.ini")

seed = int(config["boston_housing"].get("seed"))
lower_quantile = float(config["boston_housing"].get("lower_quantile"))
upper_quantile = float(config["boston_housing"].get("upper_quantile"))
# simulate_size = int(config["simulate"].get("simulate_size"))
# variation_ratio = float(config["simulate"].get("variation_ratio"))
# train_size_ratio = float(config["simulate"].get("train_size_ratio"))
# func = config["simulate"].get("func")
# model_type = config["simulate"].get("model_type")
x1_step = int(config["boston_housing"].get("x1_step"))
x2_step = int(config["boston_housing"].get("x2_step"))


opt_type = config["simulate"].get("opt_type")
max_iter = int(config["boston_housing"].get("max_iter"))
size_pop = int(config["boston_housing"].get("size_pop"))

train_df = pd.read_csv("boston_housing/data/train.csv")
test_df = pd.read_csv("boston_housing/data/test.csv")
train_df = train_df.loc[train_df["2ndFlrSF"] > 0, :]
test_df = test_df.loc[test_df["2ndFlrSF"] > 0, :]
# %%
test_df.shape
# %%
x_min = [
    train_df["1stFlrSF"].quantile(lower_quantile),
    train_df["2ndFlrSF"].quantile(lower_quantile),
]

x_max = [
    train_df["1stFlrSF"].quantile(upper_quantile),
    train_df["2ndFlrSF"].quantile(upper_quantile),
]
# %%
plt.plot(train_df["1stFlrSF"], train_df["2ndFlrSF"], "o")
plt.plot(test_df["1stFlrSF"], test_df["2ndFlrSF"], "o")
plt.vlines(
    x=train_df["1stFlrSF"].quantile(lower_quantile),
    ymin=x_min[1],
    ymax=x_max[1],
    colors="red",
    linestyles="--",
)
plt.vlines(
    x=train_df["1stFlrSF"].quantile(upper_quantile),
    ymin=x_min[1],
    ymax=x_max[1],
    colors="red",
    linestyles="--",
)

plt.hlines(
    y=train_df["2ndFlrSF"].quantile(lower_quantile),
    xmin=x_min[0],
    xmax=x_max[0],
    colors="red",
    linestyles="--",
)
plt.hlines(
    y=train_df["2ndFlrSF"].quantile(upper_quantile),
    xmin=x_min[0],
    xmax=x_max[0],
    colors="red",
    linestyles="--",
)
plt.show()

# %%
xgb = XGBRegressor(random_state=seed)
xgb.fit(train_df[["1stFlrSF", "2ndFlrSF"]], train_df["SalePrice"])

# %%
opt, a, b = optimize_f_hat(
    obj_func=xgb.predict,
    constraint_ueq=[
        lambda x: 2000 - x[0] - x[1],
    ],
    max_iter=max_iter,
    size_pop=size_pop,
    x_min=x_min,
    x_max=x_max,
    opt_type=opt_type,
)
# %%
print(opt.best_x)
print(opt.best_y)
# %%
plot_obj_surface(
    pso_opt=opt,
    f_hat=xgb,
    max_iter=max_iter,
    x_max=x_max,
    x_min=x_min,
    x1_step=x1_step,
    x2_step=x2_step,
    animate=True,
)
# %%
