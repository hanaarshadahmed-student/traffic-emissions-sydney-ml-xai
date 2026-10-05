"""
Random Forest Regressor
"""

from sklearn.ensemble import RandomForestRegressor

NAME = "random_forest"


def build_model(**params) -> RandomForestRegressor:
    return RandomForestRegressor(**params)
