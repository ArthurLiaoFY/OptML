# %%
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from simulate.optimize_response import optimize_f_hat

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
plt.plot(train_df["1stFlrSF"], train_df["2ndFlrSF"], "o")
plt.plot(test_df["1stFlrSF"], test_df["2ndFlrSF"], "o")
plt.vlines(
    x=train_df["1stFlrSF"].quantile(lower_quantile),
    ymin=train_df["2ndFlrSF"].quantile(lower_quantile),
    ymax=train_df["2ndFlrSF"].quantile(upper_quantile),
    colors="red",
    linestyles="--",
)
plt.vlines(
    x=train_df["1stFlrSF"].quantile(upper_quantile),
    ymin=train_df["2ndFlrSF"].quantile(lower_quantile),
    ymax=train_df["2ndFlrSF"].quantile(upper_quantile),
    colors="red",
    linestyles="--",
)

plt.hlines(
    y=train_df["2ndFlrSF"].quantile(lower_quantile),
    xmin=train_df["1stFlrSF"].quantile(lower_quantile),
    xmax=train_df["1stFlrSF"].quantile(upper_quantile),
    colors="red",
    linestyles="--",
)
plt.hlines(
    y=train_df["2ndFlrSF"].quantile(upper_quantile),
    xmin=train_df["1stFlrSF"].quantile(lower_quantile),
    xmax=train_df["1stFlrSF"].quantile(upper_quantile),
    colors="red",
    linestyles="--",
)
plt.show()

# %%
xgb = XGBRegressor(random_state=seed)
xgb.fit(train_df[["1stFlrSF", "2ndFlrSF"]], train_df["SalePrice"])
# %%
# x1_grid, x2_grid = np.meshgrid(
#     np.linspace(
#         start=train_df["1stFlrSF"].quantile(lower_quantile),
#         stop=train_df["1stFlrSF"].quantile(upper_quantile),
#         num=x1_step,
#     ),
#     np.linspace(
#         start=train_df["2ndFlrSF"].quantile(lower_quantile),
#         stop=train_df["2ndFlrSF"].quantile(upper_quantile),
#         num=x2_step,
#     ),
# )
# estimated_surface = xgb.predict(
#     np.array([x1_grid.reshape(-1), x2_grid.reshape(-1)]).T
# ).reshape(len(x1_grid), len(x2_grid))
# %%
# plt.contour(
#     x1_grid,
#     x2_grid,
#     estimated_surface,
#     cmap=plt.cm.coolwarm,
# )
# plt.show()
# %%
# fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
# surface = ax.plot_surface(
#     x1_grid,
#     x2_grid,
#     estimated_surface,
#     cmap=plt.cm.coolwarm,
# )

# plt.show()

# %%
opt, a, b = optimize_f_hat(
    obj_func=xgb.predict,
    constraint_ueq=[
        lambda x: 2000 - x[0] - x[1],
    ],
    max_iter=max_iter,
    size_pop=size_pop,
    x_min=[
        train_df["1stFlrSF"].quantile(lower_quantile),
        train_df["2ndFlrSF"].quantile(lower_quantile),
    ],
    x_max=[
        train_df["1stFlrSF"].quantile(upper_quantile),
        train_df["2ndFlrSF"].quantile(upper_quantile),
    ],
    opt_type=opt_type,
)
# %%
opt.best_x
# %%
opt.best_y
# %%
