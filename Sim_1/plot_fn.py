import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_obj_surface(
    x_max: list, x_min: list, f, f_hat=None, delta: float = 0.01, animate: bool = False
):
    x1_grid = np.arange(x_min[0], x_max[0], delta)
    x2_grid = np.arange(x_min[1], x_max[1], delta)
    X, Y = np.meshgrid(
        x1_grid,
        x2_grid,
    )
    Z = f(X, Y)
    Z_hat = f_hat.predict(np.array([X.reshape(-1), Y.reshape(-1)]).T).reshape(
        len(x1_grid), len(x2_grid)
    )

    fig = plt.figure(figsize=(15, 5))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")

    ax0.set_aspect("auto")
    ax0.plot_surface(
        X,
        Y,
        Z_hat,
        cmap=plt.cm.coolwarm,
    )
    ax0.set_xlabel(r"x1")
    ax0.set_ylabel(r"x2")
    ax0.set_title(
        f"Simulated Surface",
        fontweight="bold",
        fontsize=15,
    )

    ax1.contour(X, Y, Z_hat, cmap=plt.cm.coolwarm)
    ax1.set_title(
        "Contour Plot",
        fontweight="bold",
        fontsize=15,
    )
    ax1.set_xlabel(r"x1")
    ax1.set_ylabel(r"x2")

    ax2.set_aspect("auto")
    ax2.plot_surface(
        X,
        Y,
        Z,
        cmap=plt.cm.coolwarm,
    )
    ax2.set_xlabel(r"x1")
    ax2.set_ylabel(r"x2")
    ax2.set_title(
        f"Real Surface",
        fontweight="bold",
        fontsize=15,
    )
    if animate:

        def update(frame):
            ax0.view_init(elev=20.0, azim=frame)
            ax2.view_init(elev=20.0, azim=frame)
            return ax0, ax2

        ani = FuncAnimation(fig, update, frames=range(0, 360), interval=50)

    plt.tight_layout()
    plt.show()
