"""
Model registry.

Maps the model names used in config/models_config.yaml to each model's
build_model() function. src/06_run_models.py imports REGISTRY from here
and never needs to know the individual model files exist.

Only random_forest is built so far -- decision_tree, svr, and
xgboost_model follow the same pattern once it's confirmed this one
works end to end.

To add a new model:
1. Create models/<your_model>.py with a NAME string and a
   build_model(**params) function that returns an unfitted,
   scikit-learn-compatible estimator (anything with .fit()/.predict()).
2. Import it below and add it to REGISTRY.
3. Add a matching models.<name> block to config/models_config.yaml.

Nothing in src/06_run_models.py needs to change.
"""

from . import random_forest

REGISTRY = {
    random_forest.NAME: random_forest.build_model,
}
