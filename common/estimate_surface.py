import numpy as np
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


class EstimateSurface:
    def __init__(
        self,
        seed: int = 1122,
        model_type: str = "XGB",
    ) -> None:
        match model_type.upper():
            case "SVM":
                self.model = SVR()
            case "XGB":
                self.model = XGBRegressor(random_state=seed)
            case "LGBM":
                self.model = LGBMRegressor(random_state=seed)

    def fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def pred_model(self, valid_X: np.ndarray) -> np.ndarray:
        return self.model.predict(valid_X)
