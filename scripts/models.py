"""
Model registry.

Maps each model name used in config/config.yaml (training.models) to its
estimator class. 06_run_models.py and the tuning scripts build models
through this dict, so every model is loaded, trained and scored the same
way -- this file only says WHICH estimator a name means.

To add a model:
  1. Import its class below and add one line to REGISTRY.
  2. Add a matching block under training.models in config/config.yaml.
Any scikit-learn-compatible estimator (.fit()/.predict()) works.

Notes on individual models:
  ridge -- a diagnostic baseline, not a competitor for the final model.
    If the trees score a negative R2 on the "exogenous" feature set,
    compare against Ridge (which can't memorise per-station quirks):
      * Ridge also negative / near zero -> little learnable signal in
        traffic/weather alone (not a modelling-technique problem).
      * Ridge clearly better -> the trees are overfitting (most likely
        the very-short-history stations), not useless features.
    05_train_test_split.py scales continuous features and leaves 0/1
    columns unscaled, so Ridge (and SVR) work out of the box.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

REGISTRY = {
    "random_forest": RandomForestRegressor,
    "decision_tree": DecisionTreeRegressor,
    "ridge": Ridge,
    "svr": SVR,
    "xgboost": XGBRegressor,
}


def build_model(name: str, **params):
    """Unfitted estimator for a config model name, e.g.
    build_model("random_forest", n_estimators=300)."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Known models: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
