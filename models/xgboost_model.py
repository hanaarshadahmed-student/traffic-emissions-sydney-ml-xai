"""XGBoost Regressor."""

from xgboost import XGBRegressor

NAME = "xgboost"


def build_model(**params) -> XGBRegressor:
    return XGBRegressor(**params)
