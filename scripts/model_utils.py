"""
CO2/NO2 Traffic-Emissions Capstone -- Shared modeling utilities

06_run_models.py and the tuning scripts (scripts/tuning/) import from here
to load data, score predictions, and save results -- so every model in
scripts/models.py trains and is evaluated the exact same way.

Usage:

    from model_utils import load_split, get_feature_columns, evaluate, save_result

    X_train, y_train, w_train = load_split("daily", "train", feature_set="exogenous", with_weight=True)
    X_val, y_val = load_split("daily", "val", feature_set="exogenous")

    model = ...  # fit on X_train, y_train
    fit_with_optional_weight(model, X_train, y_train, w_train)
    y_pred = model.predict(X_val)

    metrics = evaluate(y_val, y_pred)
    save_result("random_forest", "daily", "exogenous", metrics)
"""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts import paths

SPLITS_DIR = paths.SPLITS_DIR
RESULTS_DIR = paths.RESULTS_DIR
MANIFEST_PATH = paths.FEATURE_MANIFEST_PATH
SPLIT_MANIFEST_PATH = paths.SPLIT_MANIFEST_PATH

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
        features = manifest["exogenous_features"]
    elif feature_set == "autoregressive":
        features = manifest["autoregressive_features"]
    elif feature_set == "all":
        features = manifest["all_candidate_features"]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")
    # Drop the near-duplicate / constant features 05_train_test_split.py
    # pruned on train rows (listed with reasons in split_manifest.json).
    excluded = load_excluded_features(grain)
    return [f for f in features if f not in excluded]


def load_excluded_features(grain: str) -> set[str]:
    if not SPLIT_MANIFEST_PATH.exists():
        return set()
    with open(SPLIT_MANIFEST_PATH) as f:
        return set(json.load(f).get(grain, {}).get("excluded_features", {}))


def load_split(
    grain: str,
    split: str,
    feature_set: str = "exogenous",
    target: str = DEFAULT_TARGET,
    with_weight: bool = False,
):
    """Load one split (train/val/test) of one grain (daily/hourly) and
    return (X, y) ready to hand straight to a model's .fit()/.predict().

    Columns are already scaled (continuous features) or left as clean
    0/1 dummies (categoricals) by 05_train_test_split.py -- nothing
    further to do here.

    with_weight=True also returns the per-row aq_quality_weight column
    (0.5 for REVIEW-flagged stations, 1.0 otherwise -- see
    03_data_preprocessing.py) as a third value, so a noisy/poorly
    AQ-matched station counts for less during TRAINING. Only pass
    with_weight=True when loading the train split -- weighting an eval
    split would score the model on a distorted version of the target
    population instead of the real one, so evaluate() always sees
    every row equally regardless of this flag.
    """
    path = SPLITS_DIR / f"{grain}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python src/05_train_test_split.py --grain {grain}` first."
        )
    df = pd.read_csv(path, dtype={"station_id": str}, low_memory=False)
    feature_cols = get_feature_columns(grain, feature_set)
    required = feature_cols + [target]
    if with_weight:
        weight_col = load_manifest()[grain]["sample_weight_column"][0]
        required = required + [weight_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from {path.name}: {missing}")
    if with_weight:
        return df[feature_cols], df[target], df[weight_col]
    return df[feature_cols], df[target]


def fit_with_optional_weight(model, X, y, sample_weight) -> bool:
    """Fits `model` with sample_weight if its .fit() accepts that
    parameter (RandomForest/DecisionTree/Ridge/SVR/XGBoost all do),
    otherwise falls back to a plain .fit(X, y) so a future model that
    doesn't support weighting still runs instead of crashing. Returns
    True if the weight was actually used, so the caller can report it.
    """
    accepts_weight = "sample_weight" in inspect.signature(model.fit).parameters
    if accepts_weight:
        model.fit(X, y, sample_weight=sample_weight)
        return True
    model.fit(X, y)
    return False


def load_station_ids(grain: str, split: str) -> pd.Series:
    """Station id for each row of the given split, in the same row order
    load_split() reads for the same (grain, split) -- so the two can be
    zipped together to break a model's predictions down per station
    without needing station_id in the model's own feature set."""
    path = SPLITS_DIR / f"{grain}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python src/05_train_test_split.py --grain {grain}` first."
        )
    return pd.read_csv(path, usecols=["station_id"], dtype={"station_id": str})["station_id"]


def evaluate_per_station(y_true, y_pred, station_ids: pd.Series) -> pd.DataFrame:
    """Same RMSE/MAE/R2 as evaluate(), broken down by station -- sorted
    worst-R2-first. Useful for telling apart "the features genuinely
    don't generalize" from "one or two short-history stations (e.g.
    6178-PR at ~60 rows total) are dragging the aggregate score down" --
    see the ridge note in scripts/models.py for why that distinction matters.

    Stations with fewer than 5 eval rows get R2 = NaN rather than a
    number: R2 computed on 2-4 points is noise, not a real error metric,
    and would be misleading sorted in among the real ones.
    """
    frame = pd.DataFrame({
        "station_id": np.asarray(station_ids),
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    rows = []
    for station_id, group in frame.groupby("station_id"):
        n = len(group)
        rows.append({
            "station_id": station_id,
            "n_rows": n,
            "rmse": float(np.sqrt(mean_squared_error(group["y_true"], group["y_pred"]))),
            "mae": float(mean_absolute_error(group["y_true"], group["y_pred"])),
            "r2": float(r2_score(group["y_true"], group["y_pred"])) if n >= 5 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("r2", na_position="first").reset_index(drop=True)


def save_per_station(
    model_name: str, grain: str, feature_set: str, split: str, per_station: pd.DataFrame,
    stage: str = "default",
) -> Path:
    """Writes one run's per-station breakdown to its own CSV (one file
    per model/grain/feature_set/split, overwritten on rerun) alongside
    the aggregate results.json, so you can open it directly to see which
    stations are driving a low aggregate score."""
    out_dir = paths.PER_STATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}_{stage}_{grain}_{feature_set}_{split}.csv"
    per_station.to_csv(path, index=False)
    return path


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


def _row_stage(row: dict) -> str:
    """Stage of a results.json row. Rows written before the `stage` column
    existed are inferred: the fine-tuning scripts always set tuned=True."""
    stage = row.get("stage") or ("tuned" if row.get("tuned") else "default")
    # "baseline" was the old name for the untuned stage -- renamed to
    # "default" so it can't be confused with the naive baseline METHODS
    return "default" if stage == "baseline" else stage


def save_result(
    model_name: str,
    grain: str,
    feature_set: str,
    metrics: dict,
    split: str = "val",
    stage: str = "default",
) -> Path:
    """Upserts one run's metrics into results/results.json,
    keyed on (model, stage, grain, feature_set, split).

    `stage` separates a model's untuned run with the config's settings
    (stage="default", 06_run_models.py)
    from its tuned run (scripts/tuning/*), so tuning XGBoost no longer
    overwrites the XGBoost default row -- both are kept, and
    07_evaluation.py shows them side by side.

    When run through run_pipeline.py the row also gets a `run_id`, linking
    it to results/runs/<run_id>/ (config, manifests, log)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = paths.RESULTS_PATH
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    results = [
        r
        for r in results
        if not (
            r["model"] == model_name
            and _row_stage(r) == stage
            and r["grain"] == grain
            and r["feature_set"] == feature_set
            and r["split"] == split
        )
    ]
    row = {
        "model": model_name,
        "stage": stage,
        "grain": grain,
        "feature_set": feature_set,
        "split": split,
        **metrics,
    }
    # Tag the row with the pipeline run that wrote it (see scripts/run_record.py),
    # so every score can be traced back to its saved config.
    run_id = os.environ.get("PIPELINE_RUN_ID")
    if run_id:
        row["run_id"] = run_id
    results.append(row)
    results_path.write_text(json.dumps(results, indent=2, default=str))
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