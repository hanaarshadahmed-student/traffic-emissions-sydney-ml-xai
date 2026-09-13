"""
CO2/NO2 Traffic-Emissions Capstone -- Model runner

Reads config/models_config.yaml and trains + evaluates every model marked
enabled: true, for every feature_set listed, on the configured grain and
split. Every model is loaded, trained, and scored the exact same way --
that logic lives here and in model_utils.py, once, so an individual model
file under models/ only has to define the estimator itself.

Results are appended to data/processed/model_results/results.json via
model_utils.save_result(), keyed on (model, grain, feature_set, split) --
re-running after a config change overwrites just that model's row, it
doesn't duplicate or wipe anyone else's.

To turn a model on/off, change its hyperparameters, or switch grain /
feature sets / split: edit config/models_config.yaml. Nothing here needs
to change for that.

To add a new model entirely: see models/__init__.py.

Usage:
    python src/06_run_models.py
    python src/06_run_models.py --config config/my_experiment.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "models_config.yaml"

# models/ sits at the repo root, alongside src/ -- not underneath it -- so
# it isn't on sys.path by default the way this script's own directory is.
sys.path.insert(0, str(ROOT_DIR))

from models import REGISTRY  # noqa: E402  (must follow the sys.path insert above)
from model_utils import Timer, evaluate, load_split, save_result  # noqa: E402


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_one(name: str, params: dict, grain: str, feature_set: str, split: str) -> dict:
    if name not in REGISTRY:
        raise KeyError(
            f"'{name}' isn't registered in models/__init__.py. "
            f"Known models: {sorted(REGISTRY)}"
        )
    X_train, y_train = load_split(grain, "train", feature_set)
    X_eval, y_eval = load_split(grain, split, feature_set)

    model = REGISTRY[name](**params)
    with Timer() as t:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    metrics = evaluate(y_eval, y_pred, train_seconds=t.seconds)
    save_result(name, grain, feature_set, metrics, split=split)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    grain = config["grain"]
    feature_sets = config["feature_sets"]
    split = config.get("split", "val")

    ran_any = False
    for name, model_config in config["models"].items():
        if not model_config.get("enabled", False):
            continue
        params = model_config.get("params") or {}
        for feature_set in feature_sets:
            metrics = run_one(name, params, grain, feature_set, split)
            ran_any = True
            print(
                f"{name:14s} | {grain} | {feature_set:14s} | {split:5s} | "
                f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
                f"R2={metrics['r2']:.4f}  ({metrics['train_seconds']}s)"
            )

    if not ran_any:
        print("No models enabled -- set enabled: true for at least one "
              f"model in {args.config}")


if __name__ == "__main__":
    main()
