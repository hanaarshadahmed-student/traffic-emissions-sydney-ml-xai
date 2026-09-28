"""Tune and train SVR on the shared scaled tabular dataset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import ParameterSampler

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "svr_tuning.yaml"
RESULTS_DIR = ROOT_DIR / "data" / "processed" / "model_results"

sys.path.insert(0, str(ROOT_DIR))

from models.svr_model import build_model  # noqa: E402
from src.model_utils import (  # noqa: E402
    Timer,
    evaluate,
    evaluate_per_station,
    fit_with_optional_weight,
    load_split,
    load_station_ids,
    save_per_station,
    save_result,
)


def load_config(path: Path) -> dict:
    with open(path) as file:
        return yaml.safe_load(file)


def build_candidates(config: dict) -> list[dict]:
    n_trials = config["n_trials"]
    sampled = []
    if n_trials > 1:
        sampled = list(
            ParameterSampler(
                config["search_space"],
                n_iter=n_trials - 1,
                random_state=config["random_state"],
            )
        )
    candidates = [config["baseline_params"], *sampled]

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = tuple(sorted(candidate.items()))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    return unique_candidates[:n_trials]


def train_candidate(
    params: dict,
    fixed_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    model_params = {**fixed_params, **params}
    model = build_model(**model_params)
    with Timer() as timer:
        weighted = fit_with_optional_weight(
            model,
            X_train,
            y_train,
            sample_weight,
        )
    predictions = model.predict(X_val)
    metrics = evaluate(y_val, predictions, train_seconds=timer.seconds)
    metrics["sample_weighted"] = weighted
    return model, predictions, metrics, model_params


def save_model_outputs(
    model,
    grain: str,
    feature_set: str,
    split: str,
    metrics: dict,
    predictions,
    y_val: pd.Series,
    X_train: pd.DataFrame,
    model_params: dict,
    n_trials: int,
    tuning_seconds: float,
) -> dict:
    model_dir = RESULTS_DIR / "trained_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"svr_{grain}_{feature_set}.joblib"
    joblib.dump(model, model_path)

    per_station = evaluate_per_station(
        y_val,
        predictions,
        load_station_ids(grain, split),
    )
    save_per_station("svr", grain, feature_set, split, per_station)

    result_metrics = {
        **metrics,
        "tuned": True,
        "n_trials": n_trials,
        "tuning_seconds": round(tuning_seconds, 3),
        "n_support_vectors": int(len(model.support_)),
        "best_params": model_params,
    }
    save_result("svr", grain, feature_set, result_metrics, split=split)

    return {
        "model": "svr",
        "grain": grain,
        "feature_set": feature_set,
        "split": split,
        "n_features": X_train.shape[1],
        "train_rows": X_train.shape[0],
        "validation_rows": len(y_val),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "n_trials": n_trials,
        "n_support_vectors": int(len(model.support_)),
        "tuning_seconds": round(tuning_seconds, 3),
        "model_path": str(model_path.relative_to(ROOT_DIR)),
        "best_kernel": model_params["kernel"],
        "best_C": model_params["C"],
        "best_epsilon": model_params["epsilon"],
        "best_gamma": model_params["gamma"],
    }


def run_experiment(
    grain: str,
    feature_set: str,
    config: dict,
    candidates: list[dict],
) -> tuple[dict, list[dict]]:
    split = config["split"]
    selection_metric = config["selection_metric"]
    if selection_metric not in {"rmse", "mae"}:
        raise ValueError("selection_metric must be 'rmse' or 'mae'")

    X_train, y_train, sample_weight = load_split(
        grain,
        "train",
        feature_set,
        with_weight=True,
    )
    X_val, y_val = load_split(grain, split, feature_set)

    trials = []
    best_run = None
    tuning_start = time.perf_counter()

    for trial_number, params in enumerate(candidates, start=1):
        model, predictions, metrics, model_params = train_candidate(
            params,
            config["fixed_params"],
            X_train,
            y_train,
            sample_weight,
            X_val,
            y_val,
        )
        trial = {
            "grain": grain,
            "feature_set": feature_set,
            "split": split,
            "trial": trial_number,
            **params,
            **metrics,
        }
        trials.append(trial)

        print(
            f"{grain:6s} | {feature_set:14s} | "
            f"trial {trial_number:02d}/{len(candidates):02d} | "
            f"RMSE={metrics['rmse']:.4f} | "
            f"MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f}",
            flush=True,
        )

        if (
            best_run is None
            or metrics[selection_metric]
            < best_run["metrics"][selection_metric]
        ):
            best_run = {
                "model": model,
                "predictions": predictions,
                "metrics": metrics,
                "model_params": model_params,
            }

    tuning_seconds = time.perf_counter() - tuning_start
    summary = save_model_outputs(
        best_run["model"],
        grain,
        feature_set,
        split,
        best_run["metrics"],
        best_run["predictions"],
        y_val,
        X_train,
        best_run["model_params"],
        len(candidates),
        tuning_seconds,
    )
    return summary, trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    candidates = build_candidates(config)
    summaries = []
    trials = []

    for grain in config["grains"]:
        for feature_set in config["feature_sets"]:
            summary, experiment_trials = run_experiment(
                grain,
                feature_set,
                config,
                candidates,
            )
            summaries.append(summary)
            trials.extend(experiment_trials)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries).sort_values(["grain", "rmse"])
    trials_frame = pd.DataFrame(trials).sort_values(
        ["grain", "feature_set", "rmse"]
    )
    summary_frame.to_csv(RESULTS_DIR / "svr_tuned_metrics.csv", index=False)
    trials_frame.to_csv(RESULTS_DIR / "svr_tuning_trials.csv", index=False)

    display_columns = [
        "grain",
        "feature_set",
        "n_features",
        "rmse",
        "mae",
        "r2",
        "best_kernel",
        "best_C",
        "best_epsilon",
        "best_gamma",
        "n_support_vectors",
    ]
    print("\nBest SVR validation results")
    print(summary_frame[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
