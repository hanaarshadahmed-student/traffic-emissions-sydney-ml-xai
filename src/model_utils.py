"""
CO2/NO2 Traffic-Emissions Capstone -- Shared modeling utilities

src/06_run_models.py imports from here to load data, score predictions,
and save results -- so every model in models/ trains and is evaluated
the exact same way. Individual model files under models/ only need to
define the estimator itself; they don't import this module directly.

Usage from 06_run_models.py:

    from model_utils import load_split, get_feature_columns, evaluate, save_result

    X_train, y_train = load_split("daily", "train", feature_set="exogenous")
    X_val, y_val = load_split("daily", "val", feature_set="exogenous")

    model = ...  # fit on X_train, y_train
    y_pred = model.predict(X_val)

    metrics = evaluate(y_val, y_pred)
    save_result("random_forest", "daily", "exogenous", metrics)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
RESULTS_DIR = PROCESSED_DIR / "model_results"
MANIFEST_PATH = PROCESSED_DIR / "feature_manifest.json"
SPLIT_MANIFEST_PATH = SPLITS_DIR / "split_manifest.json"

DEFAULT_TARGET = "no2_pphm"


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def get_feature_columns(grain: str, feature_set: str = "exogenous") -> list[str]:
    """feature_set is one of:
    - "exogenous"       -- traffic/weather/calendar/road/station only (no NO2 history)
    - "autoregressive"  -- only NO2's own lag/rolling features
    - "all"             -- exogenous + autoregressive combined

    Use "exogenous" for an explanatory model (what external factors drive
    NO2), "all" for a short-horizon forecasting model. Comparing the same
    model type across "exogenous" vs "all" is the ablation from the
    proposal's RQ2.
    """
    manifest = load_manifest()[grain]
    if feature_set == "exogenous":
        return manifest["exogenous_features"]
    elif feature_set == "autoregressive":
        return manifest["autoregressive_features"]
    elif feature_set == "all":
        return manifest["all_candidate_features"]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")


def load_split(
    grain: str,
    split: str,
    feature_set: str = "exogenous",
    target: str = DEFAULT_TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load one split (train/val/test) of one grain (daily/hourly) and
    return (X, y) ready to hand straight to a model's .fit()/.predict().

    Columns are already scaled (continuous features) or left as clean
    0/1 dummies (categoricals) by 05_train_test_split.py -- nothing
    further to do here.
    """
    path = SPLITS_DIR / f"{grain}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python src/05_train_test_split.py --grain {grain}` first."
        )
    df = pd.read_csv(path, dtype={"station_id": str}, low_memory=False)
    feature_cols = get_feature_columns(grain, feature_set)
    missing = [c for c in feature_cols + [target] if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from {path.name}: {missing}")
    return df[feature_cols], df[target]


def evaluate(y_true, y_pred, train_seconds: float | None = None) -> dict:
    """Standard metric set from the proposal's Model Evaluation section:
    RMSE, MAE, R^2, plus training time when supplied."""
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if train_seconds is not None:
        metrics["train_seconds"] = round(train_seconds, 3)
    return metrics


def save_result(model_name: str, grain: str, feature_set: str, metrics: dict, split: str = "val") -> Path:
    """Appends one run's metrics to data/processed/model_results/results.json
    so 09_evaluation.py can build the final comparison table across
    everyone's models without agreeing on a shared spreadsheet by hand."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    results = [
        r
        for r in results
        if not (
            r["model"] == model_name
            and r["grain"] == grain
            and r["feature_set"] == feature_set
            and r["split"] == split
        )
    ]
    results.append(
        {
            "model": model_name,
            "grain": grain,
            "feature_set": feature_set,
            "split": split,
            **metrics,
        }
    )
    results_path.write_text(json.dumps(results, indent=2))
    return results_path


class Timer:
    """Small context manager so a model script can time its own .fit()
    call without every script re-writing the same three lines.

        with Timer() as t:
            model.fit(X_train, y_train)
        metrics = evaluate(y_val, model.predict(X_val), train_seconds=t.seconds)
    """

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
