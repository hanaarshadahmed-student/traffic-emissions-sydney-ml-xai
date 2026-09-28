"""Support Vector Regressor."""

from sklearn.svm import SVR

NAME = "svr"


def build_model(**params) -> SVR:
    return SVR(**params)
