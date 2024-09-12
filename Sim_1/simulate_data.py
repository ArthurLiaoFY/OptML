import numpy as np
from sklearn.model_selection import train_test_split


class SimulateData:
    def __init__(
        self, seed: int = 1122, simulate_size: int = 3000, variation_ratio: float = 0.01
    ) -> None:
        self.seed = seed
        self.simulate_size = simulate_size
        self.variation_ratio = variation_ratio
        self.xy_min = [-2.0, -2.0]
        self.xy_max = [2.0, 2.0]

    def __f(self, x1, x2):
        return (-np.cos(np.pi * (x1)) * np.cos(2 * np.pi * (x2))) / (
            1 + np.power(x1, 2) + np.power(x2, 2)
        )

    def get_data(self, train_size_ratio: float = 0.8):
        seed = np.random.RandomState(self.seed)
        model_matrix = seed.uniform(
            low=self.xy_min,
            high=self.xy_max,
            size=(self.simulate_size, len(self.xy_max)),
        )
        y = np.array(
            [
                self.__f(x[0], x[1]) + seed.randn() * self.variation_ratio
                for x in model_matrix
            ]
        )
        return train_test_split(
            model_matrix,
            y,
            train_size=train_size_ratio,
            random_state=self.seed,
        )  # train_X, val_X, train_y, val_y
