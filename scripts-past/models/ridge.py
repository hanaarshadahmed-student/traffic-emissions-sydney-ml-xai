"""
Ridge Regression (linear model with L2 regularization)

Owner: <assign a name here>

This file defines the model ONLY -- loading data, training, evaluating,
and saving results all live in src/06_run_models.py and model_utils.py so
every model is trained/scored identically.

To tune hyperparameters: edit config/models_config.yaml under
models.ridge.params -- no code change needed for that.

Why this one's here: it's a diagnostic, not a competitor for the final
model choice. random_forest/decision_tree on the "exogenous" feature set
came out with negative R2 on val -- worse than just predicting the mean.
Ridge is a plain linear model with no ability to memorize per-station
quirks the way an unconstrained tree can, so comparing it against the
trees on "exogenous" tells you WHY that's happening:

  - Ridge also negative/near-zero  -> there's genuinely little
    learnable signal in traffic/weather alone for this generalization
    task (not a modeling-technique problem).
  - Ridge clearly better than the trees -> the trees are overfitting
    (most likely the very-short-history stations, e.g. 6178-PR at ~60
    rows total), not a sign the features themselves are useless.

05_train_test_split.py already scales continuous features and leaves
binary/one-hot columns unscaled specifically so a linear model like this
one works correctly out of the box -- nothing extra to do here.
"""

from sklearn.linear_model import Ridge

NAME = "ridge"


def build_model(**params) -> Ridge:
    return Ridge(**params)
