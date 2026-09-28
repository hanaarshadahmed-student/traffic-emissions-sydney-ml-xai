"""Shared random-search and learning-curve utilities."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ParameterSampler

from src.model_utils import evaluate, evaluate_per_station

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "data" / "processed" / "model_results"
LEARNING_CURVE_DIR = RESULTS_DIR / "learning_curves"
OVERFITTING_DIR = RESULTS_DIR / "overfitting"


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
                n_iter=n_trials * 5,
                random_state=config["random_state"],
            )
        )
    candidates = [
        config["baseline_params"],
        *config.get("required_candidates", []),
        *sampled,
    ]

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = tuple(sorted(candidate.items()))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    if len(unique_candidates) < n_trials:
        raise ValueError("Search space does not contain enough unique candidates.")
    return unique_candidates[:n_trials]


def chronological_prefix_indices(
    station_ids: pd.Series,
    fraction: float,
) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("Learning-curve fractions must be within (0, 1].")

    stations = station_ids.astype(str).to_numpy()
    selected = []
    for station_id in pd.unique(stations):
        positions = np.flatnonzero(stations == station_id)
        count = max(1, math.ceil(len(positions) * fraction))
        selected.extend(positions[:count])
    return np.asarray(sorted(selected), dtype=int)


def evaluate_generalization(
    y_train,
    train_predictions,
    y_validation,
    validation_predictions,
    thresholds: dict,
    train_seconds: float | None = None,
) -> dict:
    train_metrics = evaluate(y_train, train_predictions)
    validation_metrics = evaluate(
        y_validation,
        validation_predictions,
        train_seconds=train_seconds,
    )
    rmse_gap = validation_metrics["rmse"] - train_metrics["rmse"]
    relative_rmse_gap = rmse_gap / max(train_metrics["rmse"], 1e-12)
    r2_gap = train_metrics["r2"] - validation_metrics["r2"]

    rmse_flag = relative_rmse_gap > thresholds["relative_rmse_gap"]
    r2_flag = r2_gap > thresholds["r2_gap"]
    if rmse_flag and r2_flag:
        risk = "high"
    elif rmse_flag or r2_flag:
        risk = "moderate"
    else:
        risk = "low"

    return {
        **validation_metrics,
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "rmse_gap": rmse_gap,
        "relative_rmse_gap": relative_rmse_gap,
        "r2_gap": r2_gap,
        "overfitting_risk": risk,
    }


def save_overfitting_diagnostics(
    frame: pd.DataFrame,
    model_name: str,
) -> dict:
    OVERFITTING_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "grain",
        "feature_set",
        "train_rmse",
        "rmse",
        "rmse_gap",
        "relative_rmse_gap",
        "train_r2",
        "r2",
        "r2_gap",
        "overfitting_risk",
    ]
    diagnostic_frame = frame[columns].copy()
    csv_path = OVERFITTING_DIR / f"{model_name}_diagnostics.csv"
    plot_path = OVERFITTING_DIR / f"{model_name}_diagnostics.png"
    diagnostic_frame.to_csv(csv_path, index=False)

    labels = (
        diagnostic_frame["grain"]
        + "\n"
        + diagnostic_frame["feature_set"]
    )
    positions = np.arange(len(diagnostic_frame))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(
        positions - width / 2,
        diagnostic_frame["train_rmse"],
        width,
        label="Training",
    )
    axes[0].bar(
        positions + width / 2,
        diagnostic_frame["rmse"],
        width,
        label="Validation",
    )
    axes[0].set_title("RMSE generalization gap")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()

    axes[1].bar(
        positions - width / 2,
        diagnostic_frame["train_r2"],
        width,
        label="Training",
    )
    axes[1].bar(
        positions + width / 2,
        diagnostic_frame["r2"],
        width,
        label="Validation",
    )
    axes[1].set_title("R² generalization gap")
    axes[1].set_ylabel("R²")
    axes[1].legend()

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    return {
        "overfitting_diagnostics_path": str(csv_path.relative_to(ROOT_DIR)),
        "overfitting_diagnostics_plot": str(plot_path.relative_to(ROOT_DIR)),
    }


def _plot_overall_learning_curve(frame: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = (
        ("rmse", "RMSE", "lower is better"),
        ("mae", "MAE", "lower is better"),
        ("r2", "R²", "higher is better"),
    )
    for axis, (metric, title, subtitle) in zip(axes, metrics):
        axis.plot(
            frame["train_rows"],
            frame[f"train_{metric}"],
            marker="o",
            label="Training",
        )
        axis.plot(
            frame["train_rows"],
            frame[f"validation_{metric}"],
            marker="o",
            label="Validation",
        )
        axis.set_title(f"{title} ({subtitle})")
        axis.set_xlabel("Training rows")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_station_learning_curve(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for station_id, station_frame in frame.groupby("station_id"):
        axis.plot(
            station_frame["train_rows"],
            station_frame["r2"],
            marker="o",
            linewidth=1.3,
            label=station_id,
        )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title("Validation R² by station")
    axis.set_xlabel("Training rows")
    axis.set_ylabel("R²")
    axis.grid(alpha=0.25)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_learning_curves(
    model_name: str,
    grain: str,
    feature_set: str,
    fractions: list[float],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series,
    train_station_ids: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_station_ids: pd.Series,
    overfitting_thresholds: dict,
    fit_model: Callable,
) -> dict:
    rows = []
    station_frames = []

    for fraction in fractions:
        indices = chronological_prefix_indices(train_station_ids, fraction)
        X_subset = X_train.iloc[indices]
        y_subset = y_train.iloc[indices]
        weight_subset = sample_weight.iloc[indices]

        model = fit_model(X_subset, y_subset, weight_subset, X_val, y_val)
        validation_predictions = model.predict(X_val)
        diagnostics = evaluate_generalization(
            y_subset,
            model.predict(X_subset),
            y_val,
            validation_predictions,
            overfitting_thresholds,
        )
        rows.append(
            {
                "model": model_name,
                "grain": grain,
                "feature_set": feature_set,
                "train_fraction": fraction,
                "train_rows": len(X_subset),
                "train_rmse": diagnostics["train_rmse"],
                "train_mae": diagnostics["train_mae"],
                "train_r2": diagnostics["train_r2"],
                "validation_rmse": diagnostics["rmse"],
                "validation_mae": diagnostics["mae"],
                "validation_r2": diagnostics["r2"],
                "rmse_gap": diagnostics["rmse_gap"],
                "relative_rmse_gap": diagnostics["relative_rmse_gap"],
                "r2_gap": diagnostics["r2_gap"],
                "overfitting_risk": diagnostics["overfitting_risk"],
            }
        )

        station_metrics = evaluate_per_station(
            y_val,
            validation_predictions,
            val_station_ids,
        )
        station_metrics.insert(0, "train_rows", len(X_subset))
        station_metrics.insert(0, "train_fraction", fraction)
        station_metrics.insert(0, "feature_set", feature_set)
        station_metrics.insert(0, "grain", grain)
        station_metrics.insert(0, "model", model_name)
        station_frames.append(station_metrics)

    LEARNING_CURVE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_{grain}_{feature_set}"
    overall_frame = pd.DataFrame(rows).sort_values("train_fraction")
    station_frame = pd.concat(station_frames, ignore_index=True)

    overall_csv = LEARNING_CURVE_DIR / f"{stem}.csv"
    station_csv = LEARNING_CURVE_DIR / f"{stem}_per_station.csv"
    overall_plot = LEARNING_CURVE_DIR / f"{stem}.png"
    station_plot = LEARNING_CURVE_DIR / f"{stem}_per_station.png"

    overall_frame.to_csv(overall_csv, index=False)
    station_frame.to_csv(station_csv, index=False)
    _plot_overall_learning_curve(overall_frame, overall_plot)
    _plot_station_learning_curve(station_frame, station_plot)

    return {
        "learning_curve_path": str(overall_csv.relative_to(ROOT_DIR)),
        "station_learning_curve_path": str(station_csv.relative_to(ROOT_DIR)),
        "learning_curve_plot": str(overall_plot.relative_to(ROOT_DIR)),
        "station_learning_curve_plot": str(station_plot.relative_to(ROOT_DIR)),
    }
