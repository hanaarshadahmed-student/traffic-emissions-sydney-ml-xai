"""Tune and train SVR on the shared scaled tabular dataset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "svr_tuning.yaml"

sys.path.insert(0, str(ROOT_DIR))

from models.fine_tuning.common import (  # noqa: E402
    RESULTS_DIR,
    build_candidates,
    build_learning_curves,
    evaluate_generalization,
    load_config,
    save_overfitting_diagnostics,
)
from models.svr_model import build_model  # noqa: E402
from src.model_utils import (  # noqa: E402
    Timer,
    evaluate_per_station,
    fit_with_optional_weight,
    load_split,
    load_station_ids,
    save_per_station,
    save_result,
)


def train_candidate(
    params: dict,
    fixed_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    overfitting_thresholds: dict,
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
    metrics = evaluate_generalization(
        y_train,
        model.predict(X_train),
        y_val,
        predictions,
        overfitting_thresholds,
        train_seconds=timer.seconds,
    )
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
    learning_curve_paths: dict,
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
        **learning_curve_paths,
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
        "train_rmse": metrics["train_rmse"],
        "train_mae": metrics["train_mae"],
        "train_r2": metrics["train_r2"],
        "rmse_gap": metrics["rmse_gap"],
        "relative_rmse_gap": metrics["relative_rmse_gap"],
        "r2_gap": metrics["r2_gap"],
        "overfitting_risk": metrics["overfitting_risk"],
        "n_trials": n_trials,
        "n_support_vectors": int(len(model.support_)),
        "tuning_seconds": round(tuning_seconds, 3),
        "model_path": str(model_path.relative_to(ROOT_DIR)),
        **learning_curve_paths,
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
    train_station_ids = load_station_ids(grain, "train")
    val_station_ids = load_station_ids(grain, split)

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
            config["overfitting_thresholds"],
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
            f"MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f} | "
            f"R2 gap={metrics['r2_gap']:.4f} | "
            f"risk={metrics['overfitting_risk']}",
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

    def fit_learning_curve_model(
        X_subset,
        y_subset,
        weight_subset,
        X_validation,
        y_validation,
    ):
        del X_validation, y_validation
        model = build_model(**best_run["model_params"])
        fit_with_optional_weight(
            model,
            X_subset,
            y_subset,
            weight_subset,
        )
        return model

    learning_curve_paths = build_learning_curves(
        "svr",
        grain,
        feature_set,
        config["learning_curve_fractions"],
        X_train,
        y_train,
        sample_weight,
        train_station_ids,
        X_val,
        y_val,
        val_station_ids,
        config["overfitting_thresholds"],
        fit_learning_curve_model,
    )
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
        learning_curve_paths,
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
    save_overfitting_diagnostics(summary_frame, "svr")

    display_columns = [
        "grain",
        "feature_set",
        "n_features",
        "rmse",
        "mae",
        "r2",
        "train_r2",
        "r2_gap",
        "overfitting_risk",
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
