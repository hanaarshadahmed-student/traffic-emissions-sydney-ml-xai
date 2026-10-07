"""Tune the LSTM on the same split files, features and scoring as every other model.

What is searched (see config/tuning.yaml):
  window        how much history the LSTM sees: short / medium / long. The real
                length depends on the grain (window_lengths in tuning.yaml),
                e.g. hourly: 12 / 24 / 48 hours, daily: 7 / 14 / 28 days
  hidden_size   width of the network (bigger = more capacity, slower)
  num_layers    stacked LSTM layers
  dropout       share of units randomly switched off while training (more =
                less overfitting)
  learning_rate how big each training step is
  batch_size    rows per training step
  weight_decay  penalty on large weights (more = less overfitting)

Each trial trains a full LSTM (early-stopping on the validation split, exactly
as in the default run), so this is the SLOWEST tuning step: an hourly trial
takes roughly 1-2 minutes, so keep n_trials small while experimenting.
The winner is chosen on validation only; test is scored once afterwards.
Learning curves are skipped for the LSTM (they'd mean re-training it five more
times per feature set).

Run on its own:   python scripts/tuning/tune_lstm.py
Or as part of:    python run_pipeline.py --stage tune --models lstm
Needs PyTorch (pip install torch).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Run as a script, so put the repo root on the import path to reach scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import paths  # noqa: E402
from scripts.tuning.common import (  # noqa: E402
    ROOT_DIR,
    build_candidates,
    evaluate_generalization,
    grains_to_tune,
    load_config,
    save_overfitting_diagnostics,
)
from scripts.model_utils import (  # noqa: E402
    SPLITS_DIR,
    evaluate,
    evaluate_per_station,
    get_feature_columns,
    load_manifest,
    save_per_station,
    save_result,
)

MODEL = "lstm"
TARGET = "no2_pphm"
SPLITS = ["train", "val", "test"]
DEFAULT_CONFIG_PATH = paths.TUNING_CONFIG_PATH


def read_frames(grain: str) -> dict[str, pd.DataFrame]:
    return {
        s: pd.read_csv(SPLITS_DIR / f"{grain}_{s}.csv", dtype={"station_id": str}, low_memory=False)
        for s in SPLITS
    }


def run_experiment(grain, feature_set, config, candidates, frames, main_params):
    import torch

    from scripts.lstm import LSTMRegressor, build_sequences

    split = config["split"]
    selection_metric = config["selection_metric"]
    if selection_metric not in {"rmse", "mae"}:
        raise ValueError("selection_metric must be 'rmse' or 'mae'")

    features = get_feature_columns(grain, feature_set)
    weight_column = load_manifest()[grain]["sample_weight_column"][0]
    y_train, y_val = frames["train"][TARGET], frames[split][TARGET]
    weights = frames["train"][weight_column]
    window_lengths = config["window_lengths"][grain]
    sequences = {}  # window length -> prepared sequence data (built once, reused)

    trials, best_run = [], None
    tuning_start = time.perf_counter()
    for trial_number, params in enumerate(candidates, start=1):
        seq_len = int(window_lengths[params["window"]])
        if seq_len not in sequences:
            sequences[seq_len] = build_sequences(frames, features, grain, seq_len)
        data = sequences[seq_len]

        model_params = {k: v for k, v in params.items() if k != "window"}
        model = LSTMRegressor(seq_len=seq_len, verbose=False, **config["fixed_params"],
                              **{**main_params, **model_params})
        started = time.perf_counter()
        model.fit_sequences(data, y_train=y_train, sample_weight=weights, y_val=y_val, val_split=split)
        seconds = time.perf_counter() - started

        train_pred = model.predict_sequences(data, "train")
        val_pred = model.predict_sequences(data, split)
        metrics = evaluate_generalization(y_train, train_pred, y_val, val_pred,
                                          config["overfitting_thresholds"], train_seconds=seconds)
        trials.append({"grain": grain, "feature_set": feature_set, "split": split,
                       "trial": trial_number, **params, "seq_len": seq_len,
                       "best_epoch": model.best_epoch_, **metrics})
        shown = " ".join(f"{k}={v}" for k, v in params.items())
        print(
            f"{grain:6s} | {feature_set:14s} | trial {trial_number:02d}/{len(candidates):02d} | "
            f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f} | "
            f"R2 gap={metrics['r2_gap']:.4f} | risk={metrics['overfitting_risk']} | "
            f"{seconds:.0f}s, {model.best_epoch_} epochs | {shown} (seq_len={seq_len})",
            flush=True,
        )
        if best_run is None or metrics[selection_metric] < best_run["metrics"][selection_metric]:
            best_run = {"model": model, "data": data, "metrics": metrics, "val_pred": val_pred,
                        "params": {**params, "seq_len": seq_len}, "model_params": model_params}
    tuning_seconds = time.perf_counter() - tuning_start

    # --- the winner: save its scores on train / val / test ---------------------
    best, metrics = best_run["model"], best_run["metrics"]
    data = best_run["data"]
    paths.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = paths.SAVED_MODELS_DIR / f"{MODEL}_{grain}_{feature_set}.pt"
    torch.save(best.network_.state_dict(), model_path)

    save_result(MODEL, grain, feature_set,
                {**metrics, "tuned": True, "n_trials": len(candidates),
                 "tuning_seconds": round(tuning_seconds, 3), "best_params": best_run["params"],
                 "best_epoch": best.best_epoch_, "seq_len": best_run["params"]["seq_len"]},
                split=split, stage="tuned")
    save_per_station(MODEL, grain, feature_set, split,
                     evaluate_per_station(y_val, best_run["val_pred"], frames[split]["station_id"]),
                     stage="tuned")
    save_result(MODEL, grain, feature_set,
                {"rmse": metrics["train_rmse"], "mae": metrics["train_mae"], "r2": metrics["train_r2"]},
                split="train", stage="tuned")
    test_pred = best.predict_sequences(data, "test")
    y_test = frames["test"][TARGET]
    test_metrics = evaluate(y_test, test_pred)
    save_result(MODEL, grain, feature_set, test_metrics, split="test", stage="tuned")
    save_per_station(MODEL, grain, feature_set, "test",
                     evaluate_per_station(y_test, test_pred, frames["test"]["station_id"]), stage="tuned")

    summary = {
        "model": MODEL, "grain": grain, "feature_set": feature_set, "split": split,
        "n_features": len(features), "train_rows": len(y_train), "validation_rows": len(y_val),
        "rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"],
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "train_rmse": metrics["train_rmse"], "train_mae": metrics["train_mae"],
        "train_r2": metrics["train_r2"], "rmse_gap": metrics["rmse_gap"],
        "relative_rmse_gap": metrics["relative_rmse_gap"], "r2_gap": metrics["r2_gap"],
        "overfitting_risk": metrics["overfitting_risk"], "n_trials": len(candidates),
        "tuning_seconds": round(tuning_seconds, 3),
        "model_path": str(model_path.relative_to(ROOT_DIR)),
        "best_epoch": best.best_epoch_,
        **{f"best_{k}": v for k, v in best_run["params"].items()},
    }
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
        print("LSTM tuning skipped -- disabled in tuning.yaml, not enabled under "
              "training.models, or none of its grains are in this run.")
        return

    # the LSTM settings from config.yaml that aren't being searched (max_epochs, ...)
    import yaml
    main_path = args.main_config or paths.CONFIG_PATH
    with open(main_path) as file:
        main_params = dict((yaml.safe_load(file)["training"]["models"]["lstm"].get("params")) or {})
    # seq_len is replaced by the searched `window`; random_state comes from tuning.yaml
    for key in ("seq_len", "random_state"):
        main_params.pop(key, None)

    candidates = build_candidates(config)
    print(f"LSTM tuning: grains {grains}, {len(candidates)} trials per grain and feature set "
          f"(each trial trains a full LSTM -- this is the slow stage)", flush=True)

    summaries, trials = [], []
    for grain in grains:
        frames = read_frames(grain)
        for feature_set in config["feature_sets"]:
            summary, experiment_trials = run_experiment(grain, feature_set, config, candidates,
                                                        frames, main_params)
            summaries.append(summary)
            trials.extend(experiment_trials)

    paths.TUNING_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries).sort_values(["grain", "rmse"])
    trials_frame = pd.DataFrame(trials).sort_values(["grain", "feature_set", "rmse"])
    summary_frame.to_csv(paths.TUNING_DIR / "lstm_tuned_metrics.csv", index=False)
    trials_frame.to_csv(paths.TUNING_DIR / "lstm_tuning_trials.csv", index=False)
    save_overfitting_diagnostics(summary_frame, MODEL)

    columns = ["grain", "feature_set", "n_features", "rmse", "mae", "r2", "train_r2", "r2_gap",
               "overfitting_risk", "best_window", "best_seq_len", "best_hidden_size", "best_num_layers",
               "best_dropout", "best_learning_rate", "best_batch_size", "best_weight_decay", "best_epoch"]
    print("\nBest LSTM validation results")
    print(summary_frame[[c for c in columns if c in summary_frame]].to_string(index=False))


if __name__ == "__main__":
    main()
