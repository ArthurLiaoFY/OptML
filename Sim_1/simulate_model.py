import numpy as np
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


class SimulateModel:
    def __init__(self, seed: int = 1122, model_type: str = "xgb") -> None:
        match model_type:
            case "svm":
                self.model = SVR()
            case "xgb":
                self.model = XGBRegressor(random_state=seed)
            case "lgbm":
                self.model = LGBMRegressor(random_state=seed)

        pass

    def fit_model(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

        pass

    def pred_model(self, valid_X: np.ndarray):
        return self.model.predict(valid_X)
