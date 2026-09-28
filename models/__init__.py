"""
Model registry.

Maps the model names used in config/models_config.yaml to each model's
build_model() function. src/06_run_models.py imports REGISTRY from here
and never needs to know the individual model files exist.

random_forest, decision_tree, ridge, SVR, and XGBoost are registered now.

To add a new model:
1. Create models/<your_model>.py with a NAME string and a
   build_model(**params) function that returns an unfitted,
   scikit-learn-compatible estimator (anything with .fit()/.predict()).
2. Import it below and add it to REGISTRY.
3. Add a matching models.<name> block to config/models_config.yaml.

Nothing in src/06_run_models.py needs to change.
"""

from . import decision_tree, random_forest, ridge, svr_model, xgboost_model

REGISTRY = {
    random_forest.NAME: random_forest.build_model,
    decision_tree.NAME: decision_tree.build_model,
    ridge.NAME: ridge.build_model,
    svr_model.NAME: svr_model.build_model,
    xgboost_model.NAME: xgboost_model.build_model,
}
