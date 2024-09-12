# %% import packages
from time import time

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn.functional as F
from lightgbm import LGBMRegressor
from metrics_fn import R_square, mse
from mpl_toolkits.mplot3d import Axes3D
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from plot_fn import plot_contour, plot_obj_surface
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sko.DE import DE
from sko.GA import GA
from sko.PSO import PSO
from torch.utils import data
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from xgboost import XGBRegressor

shap.initjs()

# %%
variation_ratio = 0.01
n = 3000
seed = np.random.RandomState(1122)


def f(x1, x2):
    return (-np.cos(np.pi * (x1)) * np.cos(2 * np.pi * (x2))) / (
        1 + np.power(x1, 2) + np.power(x2, 2)
    )


xy_min = [-2, -2]
xy_max = [2, 2]

X = seed.uniform(low=xy_min, high=xy_max, size=(n, 2))
y = np.array([f(x[0], x[1]) + seed.randn() * variation_ratio for x in X])
# %%
train_X, val_X, train_y, val_y = train_test_split(
    X, y, train_size=0.8, random_state=1122
)


df_train_X = pd.DataFrame(train_X)
df_val_X = pd.DataFrame(val_X)
df_train_y = pd.DataFrame(train_y)
df_val_y = pd.DataFrame(val_y)


# %%
class Model(torch.nn.Module):
    def __init__(self, D_in=2, H=256, D_out=1, Hn=4):
        super().__init__()
        self.Hn = Hn  # Number of hidden layer
        self.activation = torch.nn.LeakyReLU()  # Activation function

        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(D_in, H), self.activation]
        )  # First hidden layer
        for i in range(self.Hn - 1):
            self.layers.extend(
                [torch.nn.Linear(H, H), self.activation]
            )  # Add hidden layer
        self.layers.append(torch.nn.Linear(H, D_out))  # Output layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# %%
y_val = (
    torch.tensor(df_val_y.values).float().to(device)
)  # Unsqueeze to match the shape of the output of our model
X_val = torch.tensor(df_val_X.values).float().to(device)

# Prepare data for batch training
y_train = (
    torch.tensor(df_train_y.values).float().to(device)
)  # Unsqueeze to match the shape of the output of our model
X_train = torch.tensor(df_train_X.values).float().to(device)
dataset = TensorDataset(
    X_train, y_train
)  # Make X,y into dataset so we can work with DataLoader which iterate our data in batch size
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define Model,Optimizer, Criterion
nn = Model().to(device)  # Define model and send to gpu
optimizer = torch.optim.SGD(
    nn.parameters(), lr=0.002, momentum=0.9, weight_decay=0.001
)  # What approach we use to minimize the gradient
criterion = torch.nn.MSELoss()  # Our loss function
# %%
train_losses = []  # Store the training loss
val_losses = []  # Store the validation loss
epochs = 500  # Number of time we go over the whole dataset

for epoch in range(epochs):
    running_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        # Forward propagation
        y_pred = nn(X)  # Make prediction by passing X to our model
        loss = criterion(y_pred, y)  # Calculate loss
        running_loss += loss.item()  # Add loss to running loss

        # Backward propagation
        optimizer.zero_grad()  # Empty the gradient (look up this function)
        loss.backward()  # Do backward propagation and calculate the gradient of loss with respect to every parameters (that require gradient)
        optimizer.step()  # Adjust parameters to minimize loss

    # Append train loss
    train_losses.append(
        running_loss / (batch + 1)
    )  # Add the average loss of this iteration to training loss

    # Check test loss
    y_pred = nn(X_val)
    val_loss = criterion(y_pred, y_val).item()
    val_losses.append(val_loss)

    print(
        "Train MSE: {:.4f} Train R-square: {:.4f} Valid MSE: {:.4f} Valid R-square: {:.4f}".format(
            train_losses[-1],
            0.0,
            # model_info['train']['R-square'][-1],
            val_losses[-1],
            0.0,
            # model_info['valid']['R-square'][-1]
        )
    )


# %%
# Plotting loss
def plot_loss(losses, axes=None, epoch_start=0):
    x = [i for i in range(1 + epoch_start, len(losses) + 1)]
    sns.lineplot(ax=axes, x=x, y=losses[epoch_start:])


def plot_epoch_loss(
    train_losses, test_losses, epoch1=0, epoch2=10, epoch3=50, epoch4=150
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    fig.suptitle("Losses against Epochs")

    axes[0][0].set_title("Epoch Start at " + str(epoch1))
    plot_loss(train_losses, axes[0][0], epoch1)
    plot_loss(test_losses, axes[0][0], epoch1)

    axes[0][1].set_title("Epoch Start at " + str(epoch2))
    plot_loss(train_losses, axes[0][1], epoch2)
    plot_loss(test_losses, axes[0][1], epoch2)

    axes[1][0].set_title("Epoch Start at " + str(epoch3))
    plot_loss(train_losses, axes[1][0], epoch3)
    plot_loss(test_losses, axes[1][0], epoch3)

    axes[1][1].set_title("Epoch Start at " + str(epoch4))
    plot_loss(train_losses, axes[1][1], epoch4)
    plot_loss(test_losses, axes[1][1], epoch4)


# %%
plot_epoch_loss(train_losses, val_losses)

# %%
lm = LinearRegression()
lgb = LGBMRegressor()
xgb = XGBRegressor()

lm.fit(train_X, train_y)
lgb.fit(train_X, train_y)
xgb.fit(train_X, train_y)

y_hat_LM = lm.predict(val_X)
y_hat_LGB = lgb.predict(val_X)
y_hat_XGB = xgb.predict(val_X)
nn.eval()
y_hat_NN = nn(X_val).detach_()
# %%

print("-- MSE")
print(mse(val_y, y_hat_LM))
print(mse(val_y, y_hat_LGB))
print(mse(val_y, y_hat_XGB))
print(torch.mean((y_val - y_hat_NN) ** 2).item())
print("-" * 30)

print("-- R square")
print(R_square(val_y, y_hat_LM))
print(R_square(val_y, y_hat_LGB))
print(R_square(val_y, y_hat_XGB))
print(
    1
    - torch.sum((y_val - y_hat_NN) ** 2).item()
    / torch.sum((y_val - y_val.mean()) ** 2).item()
)
print("-" * 30)

# plt.scatter(y_hat_LM, val_y, label='LM')
plt.scatter(y_hat_LGB, val_y, label="LGB")
plt.scatter(y_hat_XGB, val_y, label="XGB")
plt.scatter(y_hat_NN.cpu().numpy(), val_y, label="NN")
plt.axline(xy1=(0, 0), slope=1, color="r", lw=1)

plt.xlabel("y hat")
plt.ylabel("y")
plt.legend()
# %%
# LM

exp = shap.Explainer(lm, train_X)
shap_values = exp(train_X)
shap.plots.beeswarm(shap_values, max_display=20)

# LGB

exp = shap.Explainer(lgb, train_X)
shap_values = exp(train_X)
shap.plots.beeswarm(shap_values, max_display=20)

# XGB

exp = shap.Explainer(xgb, train_X)
shap_values = exp(train_X)
shap.plots.beeswarm(shap_values, max_display=20)

# NN
exp = shap.DeepExplainer(nn, X_train)
shap_values = exp.shap_values(X_train)
shap_exp_values = shap.Explanation(
    values=shap_values,
    base_values=[exp.expected_value[0] for i in range(len(X_train))],
    data=X_train.cpu().numpy(),
)

shap.plots.beeswarm(shap_exp_values, max_display=20)

# %%
# plot_contour(variation_ratio, n, f)
# plot_contour(variation_ratio, n, f, lm, 'lm')
# plot_contour(variation_ratio, n, f, lgb, 'lgb')
plot_contour(variation_ratio, n, f, xgb, "xgb")
# plot_contour(variation_ratio, n, f, nn, 'nn')
# %%
