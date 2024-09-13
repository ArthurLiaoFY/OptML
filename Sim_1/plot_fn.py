import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def plot_obj_surface(variation_ratio, n, f, f_hat=None, plt_name="Org"):
    delta = 0.01
    grid_len = len(np.arange(-1.75, 1.75, delta))
    X, Y = np.meshgrid(np.arange(-1.75, 1.75, delta), np.arange(-1.75, 1.75, delta))
    Z = f(X, Y)
    fig = plt.figure()
    # ax = fig.add_axes(Axes3D(fig))
    ax = plt.axes(projection="3d")

    ax.view_init(elev=20.0, azim=-45)
    ax.set_aspect("auto")
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    #                     cmap=cm.jet, edgecolor=None, linewidth=0.5, antialiased=False)
    if plt_name == "svm":
        surf = ax.plot_surface(
            X,
            Y,
            f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
                grid_len, grid_len
            ),
            cmap=plt.cm.coolwarm,
        )
    elif plt_name == "lgb":
        surf = ax.plot_surface(
            X,
            Y,
            f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
                grid_len, grid_len
            ),
            cmap=plt.cm.coolwarm,
        )
    elif plt_name == "xgb":
        surf = ax.plot_surface(
            X,
            Y,
            f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
                grid_len, grid_len
            ),
            cmap=plt.cm.coolwarm,
        )
    else:
        surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.xticks(np.arange(-1.75, 1.75, 0.74))
    plt.yticks(np.arange(-1.75, 1.75, 0.74))
    plt.xlabel(r"x")
    plt.ylabel(r"y")
    plt.title(
        "{f}(x,y)  (n:{n}, variation ratio:{variation_ratio})".format(
            n=n, variation_ratio=variation_ratio, f=plt_name
        ),
        fontweight="bold",
        fontsize=15,
    )
    plt.show()


def plot_contour(variation_ratio, n, f, f_hat=None, plt_name="Org"):

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
    if plt_name == "svm":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )
    elif plt_name == "lgb":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )
    elif plt_name == "xgb":
        Z = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
            grid_len, grid_len
        )
    else:
        Z = f(X, Y)

    plt.xlim(-1.75, 1.75)
    plt.ylim(-1.75, 1.75)
    contour = plt.contour(X, Y, Z, cmap=plt.cm.coolwarm)
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
    plt.axline(xy1=(0, 1), slope=4, color="black", lw=1, linestyle='--')
    plt.hlines(y=0.5, xmin=-1.75, xmax=1.75, color="black", lw=1, linestyle='--')
    plt.clabel(contour, inline=1, fontsize=8)
    plt.show()
