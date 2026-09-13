"""
Random Forest Regressor

Owner: <assign a name here>

This file defines the model ONLY -- loading data, training, evaluating,
and saving results all live in src/06_run_models.py and model_utils.py so
every model is trained/scored identically.

To tune hyperparameters: edit config/models_config.yaml under
models.random_forest.params -- no code change needed for that.

Only edit this file to change the model TYPE itself. `model.feature_importances_`
(available after fitting) is worth comparing against the SHAP results in
10_explainability.py later.
"""

from sklearn.ensemble import RandomForestRegressor

NAME = "random_forest"


def build_model(**params) -> RandomForestRegressor:
    return RandomForestRegressor(**params)
