"""
Decision Tree Regressor
"""

from sklearn.tree import DecisionTreeRegressor

NAME = "decision_tree"


def build_model(**params) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(**params)
