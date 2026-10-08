"""Tune the GRU on hourly sequence data with validation-only model selection."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import paths  # noqa: E402
from scripts.model_utils import (  # noqa: E402
    SPLITS_DIR,
    evaluate,
    evaluate_per_station,
    get_feature_columns,
    load_manifest,
    save_per_station,
    save_result,
)
from scripts.tuning.common import (  # noqa: E402
    ROOT_DIR,
    build_candidates,
    evaluate_generalization,
    grains_to_tune,
    load_config,
    save_overfitting_diagnostics,
)

MODEL = "gru"
TARGET = "no2_pphm"
SPLITS = ["train", "val", "test"]
DEFAULT_CONFIG_PATH = paths.TUNING_CONFIG_PATH


def read_frames(grain: str) -> dict[str, pd.DataFrame]:
    return {
        split: pd.read_csv(
            SPLITS_DIR / f"{grain}_{split}.csv",
            dtype={"station_id": str},
            low_memory=False,
        )
        for split in SPLITS
    }


def save_training_history(model, grain: str, feature_set: str) -> dict:
    paths.LEARNING_CURVE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{MODEL}_{grain}_{feature_set}_epochs"
    history_path = paths.LEARNING_CURVE_DIR / f"{stem}.csv"
    plot_path = paths.LEARNING_CURVE_DIR / f"{stem}.png"

    history = pd.DataFrame(model.history_)
    history.to_csv(history_path, index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["epoch"], history["train_loss"], marker="o", label="Training")
    axis.plot(history["epoch"], history["val_loss"], marker="o", label="Validation")
    axis.axvline(model.best_epoch_, color="black", linestyle="--", label="Best epoch")
    axis.set_title(f"GRU training history: {grain}, {feature_set}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Scaled mean squared error")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    return {
        "training_history_path": str(history_path.relative_to(ROOT_DIR)),
        "training_history_plot": str(plot_path.relative_to(ROOT_DIR)),
    }


def run_experiment(grain, feature_set, config, candidates, frames, main_params):
    import torch

    from scripts.gru import GRURegressor
    from scripts.lstm import build_sequences

    split = config["split"]
    selection_metric = config["selection_metric"]
    if selection_metric not in {"rmse", "mae"}:
        raise ValueError("selection_metric must be 'rmse' or 'mae'")

    features = get_feature_columns(grain, feature_set)
    weight_column = load_manifest()[grain]["sample_weight_column"][0]
    y_train = frames["train"][TARGET]
    y_val = frames[split][TARGET]
    weights = frames["train"][weight_column]
    window_lengths = config["window_lengths"][grain]
    sequences = {}

    trials = []
    best_run = None
    tuning_start = time.perf_counter()

    for trial_number, params in enumerate(candidates, start=1):
        seq_len = int(window_lengths[params["window"]])
        if seq_len not in sequences:
            sequences[seq_len] = build_sequences(frames, features, grain, seq_len)
        data = sequences[seq_len]

        model_params = {key: value for key, value in params.items() if key != "window"}
        model = GRURegressor(
            seq_len=seq_len,
            verbose=False,
            **config["fixed_params"],
            **{**main_params, **model_params},
        )
        started = time.perf_counter()
        model.fit_sequences(
            data,
            y_train=y_train,
            sample_weight=weights,
            y_val=y_val,
            val_split=split,
        )
        seconds = time.perf_counter() - started

        train_predictions = model.predict_sequences(data, "train")
        val_predictions = model.predict_sequences(data, split)
        metrics = evaluate_generalization(
            y_train,
            train_predictions,
            y_val,
            val_predictions,
            config["overfitting_thresholds"],
            train_seconds=seconds,
        )
        trials.append(
            {
                "model": MODEL,
                "grain": grain,
                "feature_set": feature_set,
                "split": split,
                "trial": trial_number,
                **params,
                "seq_len": seq_len,
                "best_epoch": model.best_epoch_,
                **metrics,
            }
        )
        shown = " ".join(f"{key}={value}" for key, value in params.items())
        print(
            f"{grain:6s} | {feature_set:14s} | trial {trial_number:02d}/{len(candidates):02d} | "
            f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | "
            f"R2={metrics['r2']:.4f} | R2 gap={metrics['r2_gap']:.4f} | "
            f"risk={metrics['overfitting_risk']} | {seconds:.0f}s, "
            f"{model.best_epoch_} epochs | {shown} (seq_len={seq_len})",
            flush=True,
        )

        if best_run is None or metrics[selection_metric] < best_run["metrics"][selection_metric]:
            best_run = {
                "model": model,
                "data": data,
                "metrics": metrics,
                "val_predictions": val_predictions,
                "params": {**params, "seq_len": seq_len},
            }

    tuning_seconds = time.perf_counter() - tuning_start
    best = best_run["model"]
    data = best_run["data"]
    metrics = best_run["metrics"]
    history_paths = save_training_history(best, grain, feature_set)

    paths.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = paths.SAVED_MODELS_DIR / f"{MODEL}_{grain}_{feature_set}.pt"
    torch.save(
        {
            "state_dict": best.network_.state_dict(),
            "model_params": {
                "n_inputs": data.features.shape[1] + 1,
                "hidden_size": best.hidden_size,
                "num_layers": best.num_layers,
                "dropout": best.dropout,
                "seq_len": best.window_length(grain),
            },
            "target_mean": best.y_mean_,
            "target_std": best.y_std_,
            "features": features,
        },
        model_path,
    )

    result_metrics = {
        **metrics,
        "tuned": True,
        "n_trials": len(candidates),
        "tuning_seconds": round(tuning_seconds, 3),
        "best_params": best_run["params"],
        "best_epoch": best.best_epoch_,
        "seq_len": best_run["params"]["seq_len"],
        "early_stopping": True,
        **history_paths,
    }
    save_result(MODEL, grain, feature_set, result_metrics, split=split, stage="tuned")
    save_per_station(
        MODEL,
        grain,
        feature_set,
        split,
        evaluate_per_station(y_val, best_run["val_predictions"], frames[split]["station_id"]),
        stage="tuned",
    )
    save_result(
        MODEL,
        grain,
        feature_set,
        {"rmse": metrics["train_rmse"], "mae": metrics["train_mae"], "r2": metrics["train_r2"]},
        split="train",
        stage="tuned",
    )

    y_test = frames["test"][TARGET]
    test_predictions = best.predict_sequences(data, "test")
    test_metrics = evaluate(y_test, test_predictions)
    save_result(MODEL, grain, feature_set, test_metrics, split="test", stage="tuned")
    save_per_station(
        MODEL,
        grain,
        feature_set,
        "test",
        evaluate_per_station(y_test, test_predictions, frames["test"]["station_id"]),
        stage="tuned",
    )

    summary = {
        "model": MODEL,
        "grain": grain,
        "feature_set": feature_set,
        "split": split,
        "n_features": len(features),
        "train_rows": len(y_train),
        "validation_rows": len(y_val),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "train_rmse": metrics["train_rmse"],
        "train_mae": metrics["train_mae"],
        "train_r2": metrics["train_r2"],
        "rmse_gap": metrics["rmse_gap"],
        "relative_rmse_gap": metrics["relative_rmse_gap"],
        "r2_gap": metrics["r2_gap"],
        "overfitting_risk": metrics["overfitting_risk"],
        "n_trials": len(candidates),
        "tuning_seconds": round(tuning_seconds, 3),
        "model_path": str(model_path.relative_to(ROOT_DIR)),
        "best_epoch": best.best_epoch_,
        **history_paths,
        **{f"best_{key}": value for key, value in best_run["params"].items()},
    }
    return summary, trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--main-config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config, MODEL)
    grains = grains_to_tune(MODEL, config, args.main_config)
    if not grains:
        print("GRU tuning skipped -- disabled or not selected for this run.")
        return

    import yaml

    main_path = args.main_config or paths.CONFIG_PATH
    with open(main_path) as file:
        main_params = dict(
            (yaml.safe_load(file)["training"]["models"][MODEL].get("params")) or {}
        )
    for key in ("seq_len", "random_state"):
        main_params.pop(key, None)

    candidates = build_candidates(config)
    print(
        f"GRU tuning: grains {grains}, {len(candidates)} trials per grain and feature set",
        flush=True,
    )

    summaries = []
    trials = []
    for grain in grains:
        frames = read_frames(grain)
        for feature_set in config["feature_sets"]:
            summary, experiment_trials = run_experiment(
                grain,
                feature_set,
                config,
                candidates,
                frames,
                main_params,
            )
            summaries.append(summary)
            trials.extend(experiment_trials)

    paths.TUNING_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries).sort_values(["grain", "rmse"])
    trials_frame = pd.DataFrame(trials).sort_values(["grain", "feature_set", "rmse"])
    summary_frame.to_csv(paths.TUNING_DIR / "gru_tuned_metrics.csv", index=False)
    trials_frame.to_csv(paths.TUNING_DIR / "gru_tuning_trials.csv", index=False)
    save_overfitting_diagnostics(summary_frame, MODEL)

    columns = [
        "grain",
        "feature_set",
        "n_features",
        "rmse",
        "mae",
        "r2",
        "test_rmse",
        "test_mae",
        "test_r2",
        "train_r2",
        "r2_gap",
        "overfitting_risk",
        "best_window",
        "best_seq_len",
        "best_hidden_size",
        "best_num_layers",
        "best_dropout",
        "best_learning_rate",
        "best_batch_size",
        "best_weight_decay",
        "best_epoch",
    ]
    print("\nBest GRU validation and final test results")
    print(summary_frame[[column for column in columns if column in summary_frame]].to_string(index=False))


if __name__ == "__main__":
    main()
