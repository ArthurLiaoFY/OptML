import numpy as np
from sklearn.model_selection import train_test_split


class SimulateData:
    def __init__(
        self,
        seed: int = 1122,
        simulate_size: int = 3000,
        variation_ratio: float = 0.01,
        func: str = "f1",
    ) -> None:
        self.seed = seed
        self.simulate_size = simulate_size
        self.variation_ratio = variation_ratio
        match func:
            case "f1":
                self.func = self.__f1
                self.x_min = [-1.75, -1.75]
                self.x_max = [1.75, 1.75]
            case "f2":
                self.func = self.__f2
                self.x_min = [-3.0, -3.0]
                self.x_max = [3.0, 3.0]
            case _:
                raise NotImplementedError

    def __f1(self, x1, x2):
        return (-np.cos(np.pi * (x1)) * np.cos(2 * np.pi * (x2))) / (
            1 + np.power(x1, 2) + np.power(x2, 2)
        )

    def __f2(self, x1, x2):
        return -1 * np.power(np.cos((x1 - 0.2) * x2), 2) + x1 * np.sin(2 * x1 + x2)

    def get_data(self, train_size_ratio: float = 0.8):
        seed = np.random.RandomState(self.seed)
        model_matrix = seed.uniform(
            low=self.x_min,
            high=self.x_max,
            size=(self.simulate_size, len(self.x_max)),
        )
        y = np.array(
            [
                self.func(x[0], x[1]) + seed.randn() * self.variation_ratio
                for x in model_matrix
            ]
        )
        return train_test_split(
            model_matrix,
            y,
            train_size=train_size_ratio,
            random_state=self.seed,
        )  # train_X, val_X, train_y, val_y
