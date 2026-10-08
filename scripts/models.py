"""
Model registry -- the ML models.

The project compares two kinds of method:

  ML MODELS (this file) -- learn from the features; switched on/off under
  training.models in config/config.yaml:
      1. ridge           linear regression with L2 regularisation
      2. decision_tree   a single regression tree
      3. random_forest   bagged ensemble of trees
      4. xgboost         gradient-boosted trees
      5. svr             support vector regression (RBF kernel)
      6. lstm            recurrent neural network over a window of recent
                         time steps (scripts/lstm.py, needs PyTorch)
      7. gru             gated recurrent unit network over hourly sequences
                         (scripts/gru.py, needs PyTorch)

  BASELINE METHODS (not ML -- defined in src/06_run_models.py, switched
  on/off under training.baselines in config/config.yaml):
      B1. naive_persistence           NO2 = the last observed value
      B2. naive_seasonal_climatology  NO2 = the station's train-period
                                      average for that month (and hour)
  Baselines learn nothing from the features; they're the minimum score
  an ML model has to beat to show it has learned something useful.

REGISTRY maps each config model name to its estimator class.
06_run_models.py and the tuning scripts build models through it, so every
model is loaded, trained and scored the same way.

To add a model:
  1. Import its class below and add one line to REGISTRY.
  2. Add a matching block under training.models in config/config.yaml.
Any scikit-learn-compatible estimator (.fit()/.predict()) works. A model
that needs sequences of time steps instead of single rows (LSTM or GRU)
sets `needs_sequences = True`; see scripts/lstm.py and scripts/gru.py.

Notes on individual models:
  ridge -- also a diagnostic. If the trees score a negative R2 on the
    "exogenous" feature set, compare against Ridge (which can't memorise
    per-station quirks):
      * Ridge also negative / near zero -> little learnable signal in
        traffic/weather alone (not a modelling-technique problem).
      * Ridge clearly better -> the trees are overfitting (most likely
        the very-short-history stations), not useless features.
    05_train_test_split.py scales continuous features and leaves 0/1
    columns unscaled, so Ridge, SVR and the LSTM work out of the box.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def _lstm(**params):
    # imported only when the LSTM is actually used, so the rest of the
    # pipeline still runs on a machine without PyTorch installed
    from scripts.lstm import LSTMRegressor
    return LSTMRegressor(**params)


def _gru(**params):
    from scripts.gru import GRURegressor
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
    """'baseline' for the naive baseline methods, 'ML' for everything else."""
    return "baseline" if name in BASELINE_METHODS else "ML"


def build_model(name: str, **params):
    """Unfitted estimator for a config model name, e.g.
    build_model("random_forest", n_estimators=300)."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Known models: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
