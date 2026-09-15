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
from model_utils import (  # noqa: E402
    Timer,
    evaluate,
    evaluate_per_station,
    fit_with_optional_weight,
    load_split,
    load_station_ids,
    save_per_station,
    save_result,
)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_one(name: str, params: dict, grain: str, feature_set: str, split: str) -> tuple[dict, "pd.DataFrame"]:
    if name not in REGISTRY:
        raise KeyError(
            f"'{name}' isn't registered in models/__init__.py. "
            f"Known models: {sorted(REGISTRY)}"
        )
    X_train, y_train, w_train = load_split(grain, "train", feature_set, with_weight=True)
    X_eval, y_eval = load_split(grain, split, feature_set)

    model = REGISTRY[name](**params)
    with Timer() as t:
        weighted = fit_with_optional_weight(model, X_train, y_train, w_train)
    y_pred = model.predict(X_eval)

    metrics = evaluate(y_eval, y_pred, train_seconds=t.seconds)
    metrics["sample_weighted"] = weighted
    save_result(name, grain, feature_set, metrics, split=split)

    per_station = evaluate_per_station(y_eval, y_pred, load_station_ids(grain, split))
    save_per_station(name, grain, feature_set, split, per_station)
    return metrics, per_station


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--worst-stations", type=int, default=3,
        help="How many lowest-R2 stations to print per run (0 to disable). "
             "Full breakdown is always saved to "
             "data/processed/model_results/per_station/ regardless.",
    )
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
            metrics, per_station = run_one(name, params, grain, feature_set, split)
            ran_any = True
            weight_tag = "w" if metrics["sample_weighted"] else " "
            print(
                f"{name:14s} | {grain} | {feature_set:14s} | {split:5s} | [{weight_tag}] "
                f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
                f"R2={metrics['r2']:.4f}  ({metrics['train_seconds']}s)"
            )
            if args.worst_stations > 0:
                worst = per_station.dropna(subset=["r2"]).head(args.worst_stations)
                if len(worst):
                    tags = " | ".join(
                        f"{row.station_id} n={row.n_rows} R2={row.r2:.2f}"
                        for row in worst.itertuples()
                    )
                    print(f"{'':14s}   worst stations ({split}): {tags}")

    if not ran_any:
        print("No models enabled -- set enabled: true for at least one "
              f"model in {args.config}")


if __name__ == "__main__":
    main()
