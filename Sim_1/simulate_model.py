import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


class SimulateModel:
    def __init__(self, seed: int = 1122, model_type: str = "xgb") -> None:
        match model_type:
            case "xgb":
                self.model = XGBRegressor(random_state=seed)
            case "lgbm":
                self.model = LGBMRegressor(random_state=seed)

        pass

    def fit_model(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

        pass

    def pred_model(self, valid_X: np.ndarray, valid_y: np.ndarray):
        self.model.predict(valid_X)
        pass
