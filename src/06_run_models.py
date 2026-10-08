"""
CO2/NO2 Traffic-Emissions Capstone -- Stage 06: Train models

Trains and scores two kinds of method, set in the `training` section of
config/config.yaml:

  A. BASELINE METHODS (not ML, training.baselines) -- simple rules that
     learn nothing from the features; the bar every ML model must clear:
       B1. naive_persistence           NO2 = the last observed value
       B2. naive_seasonal_climatology  NO2 = the station's train-period
                                       average for that month (and hour)

  B. ML MODELS (training.models, defined in scripts/models.py):
       1. ridge  2. decision_tree  3. random_forest  4. xgboost  5. svr
       6. lstm  7. gru (sequence models -- scripts/lstm.py / scripts/gru.py)

Every ML model is trained on the train split once per feature set and
scored on every split in report_splits, the exact same way -- that logic
lives here and in model_utils.py, once.

Results are upserted into results/results.json via model_utils.save_result(),
keyed on (model, stage, grain, feature_set, split). stage="default" here
(the model's default settings from config.yaml); the tuning scripts write
stage="tuned". Re-running after a config change overwrites just that
model's rows, it doesn't duplicate or wipe anyone else's.

To turn a model on/off, change its hyperparameters, or switch grain /
feature sets / split: edit config/config.yaml. To add a new model
entirely: see scripts/models.py.

Usage:
    python src/06_run_models.py
    python src/06_run_models.py --config config/my_experiment.yaml
    python src/06_run_models.py --jobs 4      # 4 models training at once

--jobs N trains N (model, grain, feature set) combinations at the same time
in separate processes -- much faster on a many-core computer, and the
results are identical. Each worker gets cpu_count // N threads for its own
parallel parts (random forest, XGBoost, PyTorch), so the cores aren't
oversubscribed. The default (--jobs 1) trains one at a time and prints the
LSTM's epoch-by-epoch progress.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import sys

# Repo root on the import path, so the shared code in scripts/ is importable
# whichever folder you run this from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402
from scripts.models import BASELINE_METHODS, REGISTRY, display_name  # noqa: E402
from scripts.model_utils import (  # noqa: E402
    SPLITS_DIR,
    Timer,
    evaluate,
    evaluate_per_station,
    fit_with_optional_weight,
    get_feature_columns,
    load_manifest,
    load_split,
    load_station_ids,
    save_per_station,
    save_result,
)

STAGE = "default"  # untuned run with the config's settings; tuning writes "tuned"
SPLITS = ["train", "val", "test"]
TARGET = "no2_pphm"


def load_config(config_path: Path) -> dict:
    """The `training` section of config.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f)["training"]


def _score_and_save(name, grain, feature_set, predictions, train_seconds, weighted, extra=None) -> dict:
    """predictions: {split: (y_true, y_pred)} -> metrics + per-station rows
    saved for every split. Shared by every ML model."""
    all_metrics = {}
    for eval_split, (y_true, y_pred) in predictions.items():
        metrics = evaluate(y_true, y_pred, train_seconds=train_seconds)
        metrics["sample_weighted"] = weighted
        metrics.update(extra or {})
        save_result(name, grain, feature_set, metrics, split=eval_split, stage=STAGE)
        per_station = evaluate_per_station(y_true, y_pred, load_station_ids(grain, eval_split))
        save_per_station(name, grain, feature_set, eval_split, per_station, stage=STAGE)
        all_metrics[eval_split] = (metrics, per_station)
    return all_metrics


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
    eval_splits = list(dict.fromkeys([*report_splits, split]))  # ordered, no dupes
    model = REGISTRY[name](**params)
    if getattr(model, "needs_sequences", False):
        return run_sequence_model(name, model, grain, feature_set, split, eval_splits)

    X_train, y_train, w_train = load_split(grain, "train", feature_set, with_weight=True)
    with Timer() as t:
        weighted = fit_with_optional_weight(model, X_train, y_train, w_train)

    predictions = {}
    for eval_split in eval_splits:
        if eval_split == "train":
            X_eval, y_eval = X_train, y_train  # scored unweighted, like every other split
        else:
            X_eval, y_eval = load_split(grain, eval_split, feature_set)
        predictions[eval_split] = (y_eval, model.predict(X_eval))
    return _score_and_save(name, grain, feature_set, predictions, t.seconds, weighted)


def run_sequence_model(name, model, grain, feature_set, split, eval_splits) -> dict:
    """Models that read a window of past time steps (LSTM and GRU). Same feature
    columns, rows, weights and scoring as the tabular models -- only the
    input shape differs. See scripts/lstm.py."""
    from scripts.lstm import build_sequences

    frames = {
        s: pd.read_csv(SPLITS_DIR / f"{grain}_{s}.csv", dtype={"station_id": str}, low_memory=False)
        for s in SPLITS
    }
    features = get_feature_columns(grain, feature_set)
    weight_column = load_manifest()[grain]["sample_weight_column"][0]
    data = build_sequences(frames, features, grain, model.window_length(grain))

    print(f"  {display_name(name)} | {grain} | {feature_set}: {model.window_length(grain)}-step windows, "
          f"{len(features)} features", flush=True)
    with Timer() as t:
        model.fit_sequences(
            data,
            y_train=frames["train"][TARGET],
            sample_weight=frames["train"][weight_column],
            y_val=frames[split][TARGET],
            val_split=split,
        )
    predictions = {s: (frames[s][TARGET], model.predict_sequences(data, s)) for s in eval_splits}
    extra = {"best_epoch": model.best_epoch_, "seq_len": model.window_length(grain)}
    return _score_and_save(name, grain, feature_set, predictions, t.seconds, True, extra)


# ---------------------------------------------------------------------------
# A. Baseline methods (not ML) -- the bar every ML model has to clear
# ---------------------------------------------------------------------------
# B1 naive_persistence:          "NO2 will be what it just was" (yesterday
#                                for daily, last hour for hourly). Falls back
#                                to the station's train mean where the
#                                previous value is missing.
# B2 naive_seasonal_climatology: the station's TRAIN-period average for that
#                                month (daily) or month-and-hour (hourly),
#                                falling back to the station mean, then the
#                                overall train mean.
# Neither is trained on val/test, and neither uses the feature manifest, so
# they're identical for every feature set -- they're written once per
# feature set only so they sit in every block of 07_evaluation.py's table.


def _read_split_frame(grain: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_DIR / f"{grain}_{split}.csv", dtype={"station_id": str}, low_memory=False).copy()
    ts = pd.to_datetime(df["timestamp"] if "timestamp" in df.columns else df["date"])
    df["_month"] = ts.dt.month
    df["_hour"] = ts.dt.hour
    return df


def run_baselines(grain: str, feature_sets: list[str], splits: list[str], enabled: list[str],
                  target: str = TARGET) -> dict:
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
            if name not in enabled:
                continue
            metrics = evaluate(df[target], y_pred)
            per_station = evaluate_per_station(df[target], y_pred, df["station_id"])
            for feature_set in feature_sets:
                save_result(name, grain, feature_set, metrics, split=split, stage=STAGE)
                save_per_station(name, grain, feature_set, split, per_station, stage=STAGE)
            results[(name, split)] = metrics
    return results


def grains_from(config: dict) -> list[str]:
    """training.grain may be "daily", "hourly" or a list of both."""
    grain = config.get("grain", "daily")
    return list(grain) if isinstance(grain, (list, tuple)) else [grain]


def model_grains(model_config: dict, all_grains: list[str]) -> list[str]:
    """A model can be limited to some grains with `grains: [daily]` (e.g. SVR,
    which is far too slow on hourly data)."""
    allowed = model_config.get("grains")
    return [g for g in all_grains if allowed is None or g in allowed]


def run_baseline_methods(grain: str, config: dict) -> None:
    feature_sets = config["feature_sets"]
    split = config.get("split", "val")
    report_splits = config.get("report_splits", ["train", "val", "test"])
    baselines = config.get("baselines", True)
    if isinstance(baselines, dict):  # {naive_persistence: true, ...}
        enabled_baselines = [b for b in BASELINE_METHODS if baselines.get(b, False)]
    else:                            # older configs: `baselines: true/false`
        enabled_baselines = list(BASELINE_METHODS) if baselines else []

    print(f"\n{'#' * 70}\n# {grain.upper()} data\n{'#' * 70}")
    print(f"\n=== A. Baseline methods (not ML) -- {grain} ===")
    if not enabled_baselines:
        print("  (none enabled under training.baselines)")
        return
    baseline_results = run_baselines(
        grain, feature_sets, list(dict.fromkeys([*report_splits, split])), enabled_baselines
    )
    for name in enabled_baselines:
        r2s = "  ".join(
            f"{sp}_R2={m['r2']:.4f}" for (n, sp), m in baseline_results.items() if n == name
        )
        print(f"{display_name(name):26s} | {grain} | (no features)  | {r2s}")


# Rough relative cost of each model, so the slowest start first when
# several run at once (hourly is ~18x bigger than daily).
_COST = {"lstm": 6, "gru": 6, "svr": 5, "random_forest": 3, "xgboost": 2, "decision_tree": 1, "ridge": 1}


def collect_tasks(config: dict) -> list[tuple[str, dict, str, str]]:
    """Every (model, params, grain, feature_set) to train, in config order."""
    tasks = []
    for grain in grains_from(config):
        for name, model_config in config["models"].items():
            model_config = model_config or {}
            if not model_config.get("enabled", False):
                continue
            if grain not in model_grains(model_config, [grain]):
                print(f"{display_name(name):26s} | {grain} | skipped (grains: {model_config.get('grains')})")
                continue
            for feature_set in config["feature_sets"]:
                tasks.append((name, model_config.get("params") or {}, grain, feature_set))
    return tasks


def params_for_worker(params: dict, threads: int | None) -> dict:
    """With --jobs > 1, give each worker its share of the cores and keep the
    sequence models quiet (epoch lines from several models would interleave)."""
    if threads is None:
        return params
    params = dict(params)
    if "n_jobs" in params:
        params["n_jobs"] = threads
    if "seq_len" in params:
        configured_threads = int(params.get("num_threads") or threads)
        params["num_threads"] = min(threads, configured_threads)
        params["verbose"] = False
    return params


def run_task(task, split, report_splits, threads):
    """One model x grain x feature set (runs in a worker process)."""
    name, params, grain, feature_set = task
    results = run_one(name, params_for_worker(params, threads), grain, feature_set, split, report_splits)
    return task, results


def print_result(task, results, split, worst_stations) -> None:
    name, _, grain, feature_set = task
    metrics, per_station = results[split]
    weight_tag = "w" if metrics["sample_weighted"] else " "
    r2_by_split = "  ".join(f"{s}_R2={m['r2']:.4f}" for s, (m, _) in results.items())
    print(
        f"{display_name(name):26s} | {grain} | {feature_set:14s} | [{weight_tag}] "
        f"{split} RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
        f"| {r2_by_split}  ({metrics['train_seconds']}s)",
        flush=True,
    )
    if worst_stations > 0:
        worst = per_station.dropna(subset=["r2"]).head(worst_stations)
        if len(worst):
            tags = " | ".join(
                f"{row.station_id} n={row.n_rows} R2={row.r2:.2f}" for row in worst.itertuples()
            )
            print(f"{'':26s}   worst stations ({split}): {tags}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=paths.CONFIG_PATH)
    parser.add_argument(
        "--worst-stations", type=int, default=3,
        help="How many lowest-R2 stations to print per run (0 to disable). "
             "Full breakdown is always saved to "
             "results/per_station/ regardless.",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="How many models to train at the same time (default 1). Try 4 on a "
             "16-core computer. Results are the same, just faster.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    split = config.get("split", "val")
    report_splits = config.get("report_splits", ["train", "val", "test"])

    for grain in grains_from(config):
        run_baseline_methods(grain, config)

    print("\n=== B. ML models ===")
    tasks = collect_tasks(config)
    if not tasks:
        print("No ML models ran -- set enabled: true for at least one "
              f"model under training.models in {args.config}")
        return

    jobs = max(1, min(args.jobs, len(tasks)))
    if jobs == 1:
        for task in tasks:
            _, results = run_task(task, split, report_splits, None)
            print_result(task, results, split, args.worst_stations)
        return

    threads = max(1, (os.cpu_count() or 1) // jobs)
    tasks.sort(key=lambda t: _COST.get(t[0], 1) * (18 if t[2] == "hourly" else 1), reverse=True)
    print(f"Training {len(tasks)} models, {jobs} at a time, {threads} CPU threads each "
          f"(slowest first, so results appear out of order)\n", flush=True)
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_task, t, split, report_splits, threads) for t in tasks]
        try:
            for future in as_completed(futures):
                task, results = future.result()
                print_result(task, results, split, args.worst_stations)
        except BaseException:
            for future in futures:
                future.cancel()
            raise


if __name__ == "__main__":
    main()
