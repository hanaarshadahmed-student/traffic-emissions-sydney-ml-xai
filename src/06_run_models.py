"""
CO2/NO2 Traffic-Emissions Capstone -- Stage 06: Train models

Reads the `training` section of config/config.yaml and trains + evaluates
every model marked enabled: true, for every feature_set listed, on the
configured grain and split. Every model is loaded, trained, and scored the exact same way --
that logic lives here and in model_utils.py, once; scripts/models.py only
says which estimator class each model name means.

Results are appended to results/results.json via
model_utils.save_result(), keyed on (model, stage, grain, feature_set, split)
(stage="baseline" here; the fine-tuning scripts write stage="tuned") --
re-running after a config change overwrites just that model's row, it
doesn't duplicate or wipe anyone else's.

To turn a model on/off, change its hyperparameters, or switch grain /
feature sets / split: edit config/config.yaml. Nothing here needs to
change for that.

To add a new model entirely: see scripts/models.py.

Usage:
    python src/06_run_models.py
    python src/06_run_models.py --config config/my_experiment.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import sys

# Repo root on the import path, so the shared code in scripts/ is importable
# whichever folder you run this from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402
from scripts.models import REGISTRY  # noqa: E402
from scripts.model_utils import (  # noqa: E402
    SPLITS_DIR,
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
    """The `training` section of config.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f)["training"]


def run_one(
    name: str, params: dict, grain: str, feature_set: str, split: str, report_splits: list[str]
) -> dict:
    """Train once on the train split, then score that same fitted model on
    every split in report_splits (e.g. train, val, test) -- one results.json
    row per split. `split` is the one used for the console summary and
    worst-station printout (val while iterating)."""
    if name not in REGISTRY:
        raise KeyError(
            f"'{name}' isn't registered in scripts/models.py. "
            f"Known models: {sorted(REGISTRY)}"
        )
    X_train, y_train, w_train = load_split(grain, "train", feature_set, with_weight=True)

    model = REGISTRY[name](**params)
    with Timer() as t:
        weighted = fit_with_optional_weight(model, X_train, y_train, w_train)

    all_metrics = {}
    for eval_split in dict.fromkeys([*report_splits, split]):  # ordered, no dupes
        if eval_split == "train":
            X_eval, y_eval = X_train, y_train  # scored unweighted, like every other split
        else:
            X_eval, y_eval = load_split(grain, eval_split, feature_set)
        y_pred = model.predict(X_eval)
        metrics = evaluate(y_eval, y_pred, train_seconds=t.seconds)
        metrics["sample_weighted"] = weighted
        save_result(name, grain, feature_set, metrics, split=eval_split, stage="baseline")
        per_station = evaluate_per_station(y_eval, y_pred, load_station_ids(grain, eval_split))
        save_per_station(name, grain, feature_set, eval_split, per_station, stage="baseline")
        all_metrics[eval_split] = (metrics, per_station)

    return all_metrics


# ---------------------------------------------------------------------------
# Naive baselines -- the bar every real model has to clear
# ---------------------------------------------------------------------------
# persistence:         "NO2 will be what it just was" (yesterday for daily,
#                      last hour for hourly). Falls back to the station's
#                      train mean where the previous value is missing.
# seasonal_climatology: the station's TRAIN-period average for that month
#                      (daily) or that month-and-hour (hourly), falling back
#                      to the station mean, then the overall train mean.
# Neither is trained on val/test, and neither uses the feature manifest, so
# they're identical for every feature set -- they're written once per
# feature set only so they sit in every block of 07_evaluation.py's table.
BASELINE_NAMES = ["naive_persistence", "naive_seasonal_climatology"]


def _read_split_frame(grain: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_DIR / f"{grain}_{split}.csv", dtype={"station_id": str}, low_memory=False).copy()
    ts = pd.to_datetime(df["timestamp"] if "timestamp" in df.columns else df["date"])
    df["_month"] = ts.dt.month
    df["_hour"] = ts.dt.hour
    return df


def run_baselines(grain: str, feature_sets: list[str], splits: list[str], target: str = "no2_pphm") -> dict:
    train = _read_split_frame(grain, "train")
    overall = train[target].mean()
    station_mean = train.groupby("station_id")[target].mean()
    keys = ["station_id", "_month"] + (["_hour"] if grain == "hourly" else [])
    climatology = train.groupby(keys)[target].mean().rename("_clim")

    results = {}
    for split in splits:
        df = train if split == "train" else _read_split_frame(grain, split)
        fallback = df["station_id"].map(station_mean).fillna(overall)
        preds = {
            "naive_persistence": (
                df["baseline_persistence"] if "baseline_persistence" in df.columns
                else pd.Series(np.nan, index=df.index)
            ).fillna(fallback),
            "naive_seasonal_climatology": df.join(climatology, on=keys)["_clim"].fillna(fallback),
        }
        for name, y_pred in preds.items():
            metrics = evaluate(df[target], y_pred)
            per_station = evaluate_per_station(df[target], y_pred, df["station_id"])
            for feature_set in feature_sets:
                save_result(name, grain, feature_set, metrics, split=split, stage="baseline")
                save_per_station(name, grain, feature_set, split, per_station, stage="baseline")
            results[(name, split)] = metrics
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=paths.CONFIG_PATH)
    parser.add_argument(
        "--worst-stations", type=int, default=3,
        help="How many lowest-R2 stations to print per run (0 to disable). "
             "Full breakdown is always saved to "
             "results/per_station/ regardless.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    grain = config["grain"]
    feature_sets = config["feature_sets"]
    split = config.get("split", "val")
    # Every split the fitted model gets scored on and saved for. Hyperparameter
    # choices should still be made on `split` (val) only -- test is reported,
    # not tuned on.
    report_splits = config.get("report_splits", ["train", "val", "test"])

    if config.get("baselines", True):
        baseline_results = run_baselines(
            grain, feature_sets, list(dict.fromkeys([*report_splits, split]))
        )
        for name in BASELINE_NAMES:
            r2s = "  ".join(
                f"{sp}_R2={m['r2']:.4f}" for (n, sp), m in baseline_results.items() if n == name
            )
            print(f"{name:27s} | {grain} | (no features) | {r2s}")

    ran_any = False
    for name, model_config in config["models"].items():
        if not model_config.get("enabled", False):
            continue
        params = model_config.get("params") or {}
        for feature_set in feature_sets:
            results = run_one(name, params, grain, feature_set, split, report_splits)
            ran_any = True
            metrics, per_station = results[split]
            weight_tag = "w" if metrics["sample_weighted"] else " "
            r2_by_split = "  ".join(
                f"{s}_R2={m['r2']:.4f}" for s, (m, _) in results.items()
            )
            print(
                f"{name:14s} | {grain} | {feature_set:14s} | [{weight_tag}] "
                f"{split} RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
                f"| {r2_by_split}  ({metrics['train_seconds']}s)"
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
              f"model under training.models in {args.config}")


if __name__ == "__main__":
    main()