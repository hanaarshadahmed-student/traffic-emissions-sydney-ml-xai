"""Tune the decision tree on the shared scaled tabular dataset.

A single tree overfits easily: left alone it memorises the training rows. The
settings searched here all limit that --
  max_depth          how many questions deep the tree may go
  min_samples_leaf   the fewest rows allowed in a final leaf
  min_samples_split  the fewest rows needed to split a branch further
  ccp_alpha          prunes branches that add little (0 = no pruning)
The first candidates tried are the config.yaml settings, then a few hand-picked
ones, then random draws from the search space in config/tuning.yaml. The winner
is chosen on the validation split only; test is scored once afterwards.

Run on its own:   python scripts/tuning/tune_decision_tree.py
Or as part of:    python run_pipeline.py --stage tune --models decision_tree
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from joblib import Parallel, delayed

# Run as a script, so put the repo root on the import path to reach scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import paths  # noqa: E402
from scripts.tuning.common import (  # noqa: E402
    ROOT_DIR,
    build_candidates,
    build_learning_curves,
    evaluate_generalization,
    grains_to_tune,
    load_config,
    save_overfitting_diagnostics,
    save_train_and_test_results,
)
from scripts.models import build_model as _build_model  # noqa: E402
from scripts.model_utils import (  # noqa: E402
    Timer,
    evaluate_per_station,
    fit_with_optional_weight,
    load_split,
    load_station_ids,
    save_per_station,
    save_result,
)

MODEL = "decision_tree"
DEFAULT_CONFIG_PATH = paths.TUNING_CONFIG_PATH


def build_model(**params):
    return _build_model(MODEL, **params)


def train_candidate(params, fixed_params, X_train, y_train, sample_weight, X_val, y_val, thresholds):
    model_params = {**fixed_params, **params}
    model = build_model(**model_params)
    with Timer() as timer:
        weighted = fit_with_optional_weight(model, X_train, y_train, sample_weight)
    predictions = model.predict(X_val)
    metrics = evaluate_generalization(
        y_train, model.predict(X_train), y_val, predictions, thresholds,
        train_seconds=timer.seconds,
    )
    metrics["sample_weighted"] = weighted
    return model, predictions, metrics, model_params


def save_model_outputs(model, grain, feature_set, split, metrics, predictions, y_val,
                       X_train, model_params, n_trials, tuning_seconds, learning_curve_paths) -> dict:
    model_dir = paths.SAVED_MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{MODEL}_{grain}_{feature_set}.joblib"
    joblib.dump(model, model_path)

    per_station = evaluate_per_station(y_val, predictions, load_station_ids(grain, split))
    save_per_station(MODEL, grain, feature_set, split, per_station, stage="tuned")

    result_metrics = {
        **metrics,
        "tuned": True,
        "n_trials": n_trials,
        "tuning_seconds": round(tuning_seconds, 3),
        "best_params": model_params,
        **learning_curve_paths,
    }
    save_result(MODEL, grain, feature_set, result_metrics, split=split, stage="tuned")
    test_metrics = save_train_and_test_results(MODEL, model, grain, feature_set, metrics)

    return {
        "model": MODEL,
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
        "model_path": str(model_path.relative_to(ROOT_DIR)),
        **learning_curve_paths,
        "best_max_depth": model_params.get("max_depth"),
        "best_min_samples_leaf": model_params.get("min_samples_leaf"),
        "best_min_samples_split": model_params.get("min_samples_split"),
        "best_ccp_alpha": model_params.get("ccp_alpha"),
        "n_leaves": int(model.get_n_leaves()),
    }


def run_experiment(grain: str, feature_set: str, config: dict, candidates: list[dict]):
    split = config["split"]
    selection_metric = config["selection_metric"]
    if selection_metric not in {"rmse", "mae"}:
        raise ValueError("selection_metric must be 'rmse' or 'mae'")

    X_train, y_train, sample_weight = load_split(grain, "train", feature_set, with_weight=True)
    X_val, y_val = load_split(grain, split, feature_set)
    train_station_ids = load_station_ids(grain, "train")
    val_station_ids = load_station_ids(grain, split)

    trials, best_run = [], None
    tuning_start = time.perf_counter()
    # A tree fits on one core, but candidates are independent, so fit them side
    # by side (n_jobs in tuning.yaml; -1 = all cores).
    fitted = Parallel(n_jobs=config.get("n_jobs", -1))(
        delayed(train_candidate)(
            params, config["fixed_params"], X_train, y_train, sample_weight,
            X_val, y_val, config["overfitting_thresholds"],
        )
        for params in candidates
    )
    for trial_number, (params, (model, predictions, metrics, model_params)) in enumerate(
        zip(candidates, fitted), start=1
    ):
        trials.append({"grain": grain, "feature_set": feature_set, "split": split,
                       "trial": trial_number, **params, **metrics})
        shown = " ".join(f"{k}={v}" for k, v in params.items())
        print(
            f"{grain:6s} | {feature_set:14s} | trial {trial_number:02d}/{len(candidates):02d} | "
            f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f} | "
            f"R2 gap={metrics['r2_gap']:.4f} | risk={metrics['overfitting_risk']} | {shown}",
            flush=True,
        )
        if best_run is None or metrics[selection_metric] < best_run["metrics"][selection_metric]:
            best_run = {"model": model, "predictions": predictions,
                        "metrics": metrics, "model_params": model_params}
    tuning_seconds = time.perf_counter() - tuning_start

    def fit_learning_curve_model(X_subset, y_subset, weight_subset, X_validation, y_validation):
        del X_validation, y_validation
        model = build_model(**best_run["model_params"])
        fit_with_optional_weight(model, X_subset, y_subset, weight_subset)
        return model

    learning_curve_paths = build_learning_curves(
        MODEL, grain, feature_set, config["learning_curve_fractions"],
        X_train, y_train, sample_weight, train_station_ids, X_val, y_val, val_station_ids,
        config["overfitting_thresholds"], fit_learning_curve_model,
    )
    summary = save_model_outputs(
        best_run["model"], grain, feature_set, split, best_run["metrics"],
        best_run["predictions"], y_val, X_train, best_run["model_params"],
        len(candidates), tuning_seconds, learning_curve_paths,
    )
    return summary, trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Tuning config (search spaces).")
    parser.add_argument("--main-config", type=Path, default=None,
                        help="config.yaml of the pipeline run -- limits tuning to the grains "
                             "and models that run trains.")
    args = parser.parse_args()

    config = load_config(args.config, MODEL)
    grains = grains_to_tune(MODEL, config, args.main_config)
    if not grains:
        print("Decision tree tuning skipped -- disabled in tuning.yaml, not enabled under "
              "training.models, or none of its grains are in this run.")
        return
    candidates = build_candidates(config)
    print(f"Decision tree tuning: grains {grains}, {len(candidates)} trials per grain and feature set",
          flush=True)

    summaries, trials = [], []
    for grain in grains:
        for feature_set in config["feature_sets"]:
            summary, experiment_trials = run_experiment(grain, feature_set, config, candidates)
            summaries.append(summary)
            trials.extend(experiment_trials)

    paths.TUNING_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries).sort_values(["grain", "rmse"])
    trials_frame = pd.DataFrame(trials).sort_values(["grain", "feature_set", "rmse"])
    summary_frame.to_csv(paths.TUNING_DIR / "decision_tree_tuned_metrics.csv", index=False)
    trials_frame.to_csv(paths.TUNING_DIR / "decision_tree_tuning_trials.csv", index=False)
    save_overfitting_diagnostics(summary_frame, MODEL)

    display_columns = ["grain", "feature_set", "n_features", "rmse", "mae", "r2", "train_r2",
                       "r2_gap", "overfitting_risk", "best_max_depth", "best_min_samples_leaf",
                       "best_min_samples_split", "best_ccp_alpha", "n_leaves"]
    print("\nBest decision tree validation results")
    print(summary_frame[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
    