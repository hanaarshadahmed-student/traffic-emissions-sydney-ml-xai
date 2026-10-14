"""
Model registry: maps each model name in config.yaml to its estimator class, so every model is built, trained and scored the same way.

To add a model:
  1. Import its class below and add one line to REGISTRY.
  2. Add a matching block under training.models in config/config.yaml.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def _lstm(**params):
    from models.sequence_models import LSTMRegressor
    return LSTMRegressor(**params)


def _gru(**params):
    from models.sequence_models import GRURegressor
    return GRURegressor(**params)


REGISTRY = {
    "ridge": Ridge,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
    "svr": SVR,
    "lstm": _lstm,
    "gru": _gru,
}

# Not ML -- computed directly in src/06_run_models.py (see the docstring above).
BASELINE_METHODS = ["naive_persistence", "naive_seasonal_climatology"]


# How each method is labelled in printed tables and charts
DISPLAY_NAMES = {
    "naive_persistence": "B1 Persistence",
    "naive_seasonal_climatology": "B2 Seasonal climatology",
    "ridge": "1 Ridge",
    "decision_tree": "2 Decision tree",
    "random_forest": "3 Random forest",
    "xgboost": "4 XGBoost",
    "svr": "5 SVR",
    "lstm": "6 LSTM",
    "gru": "7 GRU",
}


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def model_type(name: str) -> str:
    return "baseline" if name in BASELINE_METHODS else "ML"


def build_model(name: str, **params):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Known models: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
