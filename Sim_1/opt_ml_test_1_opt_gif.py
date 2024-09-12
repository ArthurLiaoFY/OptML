# %% import packages
from time import time

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from metrics_fn import R_square, mse
from mpl_toolkits.mplot3d import Axes3D
from plot_fn import plot_contour, plot_obj_surface
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sko.DE import DE
from sko.GA import GA
from sko.PSO import PSO
from xgboost import XGBRegressor

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
print("-" * 30)

print("-- R square")
print(R_square(val_y, y_hat_LM))
print(R_square(val_y, y_hat_LGB))
print(R_square(val_y, y_hat_XGB))

print("-" * 30)

# plt.scatter(y_hat_LM, val_y, label='LM')
plt.scatter(y_hat_LGB, val_y, label="LGB")
plt.scatter(y_hat_XGB, val_y, label="XGB")
plt.axline(xy1=(0, 0), slope=1, color="r", lw=1)

plt.xlabel("y hat")
plt.ylabel("y")
plt.legend()
# %% plot obj surface
plot_obj_surface(variation_ratio, n, f)
plot_obj_surface(variation_ratio, n, f, lm, "lm")
plot_obj_surface(variation_ratio, n, f, lgb, "lgb")
plot_obj_surface(variation_ratio, n, f, xgb, "xgb")

# %% plot contour
plot_contour(variation_ratio, n, f)
plot_contour(variation_ratio, n, f, lm, "lm")
plot_contour(variation_ratio, n, f, lgb, "lgb")
plot_contour(variation_ratio, n, f, xgb, "xgb")

# %%


def obj_func_lm(p):
    x1, x2 = p
    return float(lm.predict(np.array([[x1, x2]])).item())


def obj_func_lgb(p):
    x1, x2 = p
    return float(lgb.predict(np.array([[x1, x2]])).item())


def obj_func_xgb(p):
    x1, x2 = p
    return float(xgb.predict(np.array([[x1, x2]])).item())


# %% DE
def optimize_f_hat(obj_func, f_hat, obj_fn):
    """
    min f(x1, x2) = f(x1, x2)
    s.t.
        x2 >= 0.5  => 0.5-x2 <= 0
        x2 - 4*x1 >= 1  => 1-x2+4*x1 <= 0
        -1.75 <= x1, x2 <= 1.75
    """
    constraint_ueq = [lambda x: 0.5 - x[1], lambda x: 1 - x[1] + 4 * x[0]]

    max_iter = 10000
    size_pop = 10
    # DE
    start = time()
    de = DE(
        func=obj_func,
        n_dim=2,
        size_pop=size_pop,
        max_iter=max_iter,
        lb=[-1.75, -1.75],
        ub=[1.75, 1.75],
        # constraint_eq=constraint_eq,
        constraint_ueq=constraint_ueq,
    )

    de_all_history_x1 = []
    de_all_history_x2 = []

    fig = plt.figure()
    delta = 0.01
    grid_len = len(np.arange(-1.75, 1.75, delta))

    plt.title(
        "Contour Plot (n:{n}, variation ratio:{variation_ratio})".format(
            n=n, variation_ratio=variation_ratio
        ),
        fontweight="bold",
        fontsize=15,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    X, Y = np.meshgrid(np.arange(-1.75, 1.75, delta), np.arange(-1.75, 1.75, delta))
    if obj_fn == "lm":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )
    elif obj_fn == "lgb":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )
    elif obj_fn == "xgb":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )

    else:
        Z = f(X, Y)

    plt.xlim(-1.75, 1.75)
    plt.ylim(-1.75, 1.75)
    contour = plt.contour(X, Y, Z, linewidth=5, cmap=plt.cm.coolwarm)
    min_z_idx = Z[(Y >= 0.5) & (Y - 4 * X >= 1)].argmin()
    plt.plot(
        X[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        Y[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        c="r",
        marker="o",
        ms=4,
    )
    plt.text(
        X[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        Y[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        str(round(Z[(Y >= 0.5) & (Y - 4 * X >= 1)].min(), 4))
        + " ("
        + str(round(X[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx], 4))
        + ", "
        + str(round(Y[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx], 4))
        + ")",
    )
    plt.axline(xy1=(0, 1), slope=4, color="r", lw=1)
    plt.hlines(y=0.5, xmin=-1.75, xmax=1.75, color="r", lw=1)
    plt.clabel(contour, inline=1, fontsize=8)

    (line,) = plt.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        de.run(1)
        de_all_history_x1.append(de.best_x[0])
        de_all_history_x2.append(de.best_x[1])
        line.set_data(de_all_history_x1, de_all_history_x2)

        return (line,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=500, interval=100, blit=True
    )

    anim.save(obj_fn + "_de.gif", writer="ffmpeg", fps=30)

    print("best_x:", de.best_x, "\nbest_obj_func:", de.best_y)
    print("total time cost : {time_diff}".format(time_diff=time() - start))
    # DE plot
    Y_history = pd.DataFrame(de.all_history_Y)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(Y_history.index, Y_history.values, ".", color="red")
    Y_history.min(axis=1).cummin().plot(kind="line")
    plt.title(
        "DE iteration result (n:{n}, variation ratio:{variation_ratio})".format(
            n=n, variation_ratio=variation_ratio
        )
    )
    plt.show()

    # GA
    start = time()
    ga = GA(
        func=obj_func,
        n_dim=2,
        size_pop=size_pop,
        max_iter=max_iter,
        lb=[-1.75, -1.75],
        ub=[1.75, 1.75],
        # constraint_eq=constraint_eq,
        constraint_ueq=constraint_ueq,
    )

    ga_all_history_x1 = []
    ga_all_history_x2 = []

    fig = plt.figure()
    delta = 0.01
    grid_len = len(np.arange(-1.75, 1.75, delta))

    plt.title(
        "Contour Plot (n:{n}, variation ratio:{variation_ratio})".format(
            n=n, variation_ratio=variation_ratio
        ),
        fontweight="bold",
        fontsize=15,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    X, Y = np.meshgrid(np.arange(-1.75, 1.75, delta), np.arange(-1.75, 1.75, delta))
    if obj_fn == "lm":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )
    elif obj_fn == "lgb":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )
    elif obj_fn == "xgb":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )
    else:
        Z = f(X, Y)

    plt.xlim(-1.75, 1.75)
    plt.ylim(-1.75, 1.75)
    contour = plt.contour(X, Y, Z, linewidth=5, cmap=plt.cm.coolwarm)
    min_z_idx = Z[(Y >= 0.5) & (Y - 4 * X >= 1)].argmin()
    plt.plot(
        X[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        Y[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        c="r",
        marker="o",
        ms=4,
    )
    plt.text(
        X[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        Y[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        str(round(Z[(Y >= 0.5) & (Y - 4 * X >= 1)].min(), 4))
        + " ("
        + str(round(X[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx], 4))
        + ", "
        + str(round(Y[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx], 4))
        + ")",
    )
    plt.axline(xy1=(0, 1), slope=4, color="r", lw=1)
    plt.hlines(y=0.5, xmin=-1.75, xmax=1.75, color="r", lw=1)
    plt.clabel(contour, inline=1, fontsize=8)

    (line,) = plt.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        ga.run(1)
        ga_all_history_x1.append(ga.best_x[0])
        ga_all_history_x2.append(ga.best_x[1])
        line.set_data(ga_all_history_x1, ga_all_history_x2)

        return (line,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=500, interval=100, blit=True
    )

    anim.save(obj_fn + "_ga.gif", writer="ffmpeg", fps=30)

    print("best_x:", ga.best_x, "\nbest_obj_func:", ga.best_y)
    print("total time cost : {time_diff}".format(time_diff=time() - start))
    # GA plot

    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(Y_history.index, Y_history.values, ".", color="red")
    Y_history.min(axis=1).cummin().plot(kind="line")
    plt.title(
        "GA iteration result (n:{n}, variation ratio:{variation_ratio})".format(
            n=n, variation_ratio=variation_ratio
        )
    )
    plt.show()

    # PSO
    start = time()
    pso = PSO(
        func=obj_func,
        n_dim=2,
        pop=size_pop,
        max_iter=max_iter,
        lb=[-1.75, -1.75],
        ub=[1.75, 1.75],
        # constraint_eq=constraint_eq,
        constraint_ueq=constraint_ueq,
    )
    pso_all_history_x1 = []
    pso_all_history_x2 = []
    pso_all_history_Y = []

    fig = plt.figure()
    delta = 0.01
    grid_len = len(np.arange(-1.75, 1.75, delta))

    plt.title(
        "Contour Plot (n:{n}, variation ratio:{variation_ratio})".format(
            n=n, variation_ratio=variation_ratio
        ),
        fontweight="bold",
        fontsize=15,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    X, Y = np.meshgrid(np.arange(-1.75, 1.75, delta), np.arange(-1.75, 1.75, delta))
    if obj_fn == "lm":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )
    elif obj_fn == "lgb":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )
    elif obj_fn == "xgb":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )

    else:
        Z = f(X, Y)

    plt.xlim(-1.75, 1.75)
    plt.ylim(-1.75, 1.75)
    contour = plt.contour(X, Y, Z, linewidth=5, cmap=plt.cm.coolwarm)
    min_z_idx = Z[(Y >= 0.5) & (Y - 4 * X >= 1)].argmin()
    plt.plot(
        X[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        Y[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        c="r",
        marker="o",
        ms=4,
    )
    plt.text(
        X[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        Y[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx],
        str(round(Z[(Y >= 0.5) & (Y - 4 * X >= 1)].min(), 4))
        + " ("
        + str(round(X[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx], 4))
        + ", "
        + str(round(Y[(Y >= 0.5) & (Y - 4 * X >= 1)][min_z_idx], 4))
        + ")",
    )
    plt.axline(xy1=(0, 1), slope=4, color="r", lw=1)
    plt.hlines(y=0.5, xmin=-1.75, xmax=1.75, color="r", lw=1)
    plt.clabel(contour, inline=1, fontsize=8)

    (line,) = plt.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        pso.run(1)
        pso_all_history_x1.append(pso.best_x[0])
        pso_all_history_x2.append(pso.best_x[1])
        pso_all_history_Y.append(pso.Y[:, 0])

        line.set_data(pso_all_history_x1, pso_all_history_x2)

        return (line,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=500, interval=100, blit=True
    )

    anim.save(obj_fn + "_pso.gif", writer="ffmpeg", fps=30)

    print("best_x is ", pso.best_x, "\nbest_y is", pso.best_y)
    print("total time cost : {time_diff}".format(time_diff=time() - start))
    # PSO plot
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(range(len(pso_all_history_Y)), pso_all_history_Y, ".", color="red")
    ax[1].plot(pso.gbest_y_hist)
    plt.title(
        "PSO iteration result (n:{n}, variation ratio:{variation_ratio})".format(
            n=n, variation_ratio=variation_ratio
        )
    )
    plt.show()


# %%
optimize_f_hat(obj_func_lm, lm, obj_fn="lm")
# %%
optimize_f_hat(obj_func_lgb, lgb, obj_fn="lgb")
# %%
optimize_f_hat(obj_func_xgb, xgb, obj_fn="xgb")
# %%
