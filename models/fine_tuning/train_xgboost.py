"""Tune and train XGBoost on the shared daily and hourly tabular datasets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "xgboost_tuning.yaml"

sys.path.insert(0, str(ROOT_DIR))

from models.fine_tuning.common import (  # noqa: E402
    RESULTS_DIR,
    build_candidates,
    build_learning_curves,
    evaluate_generalization,
    load_config,
    save_overfitting_diagnostics,
    save_train_and_test_results,
)
from models.xgboost_model import build_model  # noqa: E402
from src.model_utils import (  # noqa: E402
    Timer,
    evaluate_per_station,
    load_split,
    load_station_ids,
    save_per_station,
    save_result,
)


def train_candidate(
    params: dict,
    fixed_params: dict,
    random_state: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    overfitting_thresholds: dict,
):
    model_params = {
        **fixed_params,
        **params,
        "random_state": random_state,
    }
    model = build_model(**model_params)
    with Timer() as timer:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
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
    metrics["sample_weighted"] = True
    metrics["best_iteration"] = int(model.best_iteration)
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
    importance_dir = RESULTS_DIR / "feature_importance"
    model_dir.mkdir(parents=True, exist_ok=True)
    importance_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"xgboost_{grain}_{feature_set}.joblib"
    joblib.dump(model, model_path)

    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_path = importance_dir / f"xgboost_{grain}_{feature_set}.csv"
    importance.to_csv(importance_path, index=False)

    per_station = evaluate_per_station(
        y_val,
        predictions,
        load_station_ids(grain, split),
    )
    save_per_station("xgboost", grain, feature_set, split, per_station, stage="tuned")

    result_metrics = {
        **metrics,
        "tuned": True,
        "n_trials": n_trials,
        "tuning_seconds": round(tuning_seconds, 3),
        "early_stopping": True,
        "best_iteration": metrics["best_iteration"],
        "best_params": model_params,
        **learning_curve_paths,
    }
    save_result("xgboost", grain, feature_set, result_metrics, split=split, stage="tuned")
    test_metrics = save_train_and_test_results("xgboost", model, grain, feature_set, metrics)

    return {
        "model": "xgboost",
        "grain": grain,
        "feature_set": feature_set,
        "split": split,
        "n_features": X_train.shape[1],
        "train_rows": X_train.shape[0],
        "validation_rows": len(y_val),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        **test_metrics,
        "train_rmse": metrics["train_rmse"],
        "train_mae": metrics["train_mae"],
        "train_r2": metrics["train_r2"],
        "rmse_gap": metrics["rmse_gap"],
        "relative_rmse_gap": metrics["relative_rmse_gap"],
        "r2_gap": metrics["r2_gap"],
        "overfitting_risk": metrics["overfitting_risk"],
        "n_trials": n_trials,
        "tuning_seconds": round(tuning_seconds, 3),
        "early_stopping": True,
        "best_iteration": metrics["best_iteration"],
        "model_path": str(model_path.relative_to(ROOT_DIR)),
        "importance_path": str(importance_path.relative_to(ROOT_DIR)),
        **learning_curve_paths,
        **{
            f"best_{name}": value
            for name, value in model_params.items()
            if name
            in {
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
            }
        },
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
    random_state = config["random_state"]

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
            random_state,
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
            f"trees={metrics['best_iteration'] + 1} | "
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
        model = build_model(**best_run["model_params"])
        model.fit(
            X_subset,
            y_subset,
            sample_weight=weight_subset,
            eval_set=[(X_validation, y_validation)],
            verbose=False,
        )
        return model

    learning_curve_paths = build_learning_curves(
        "xgboost",
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
    summary_frame.to_csv(RESULTS_DIR / "xgboost_tuned_metrics.csv", index=False)
    trials_frame.to_csv(RESULTS_DIR / "xgboost_tuning_trials.csv", index=False)
    save_overfitting_diagnostics(summary_frame, "xgboost")

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
        "best_n_estimators",
        "best_learning_rate",
        "best_max_depth",
        "best_min_child_weight",
        "best_subsample",
        "best_colsample_bytree",
        "best_reg_alpha",
        "best_reg_lambda",
        "best_iteration",
    ]
    print("\nBest XGBoost validation results")
    print(summary_frame[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()