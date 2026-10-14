"""
Model utils: shared helpers for loading splits, scoring predictions and saving results, so every model is trained and evaluated the same way. 
Used by 06_run_models.py and the tuning scripts.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts import paths
from scripts.stations import station_label

SPLITS_DIR = paths.SPLITS_DIR
RESULTS_DIR = paths.RESULTS_DIR
MANIFEST_PATH = paths.FEATURE_MANIFEST_PATH
SPLIT_MANIFEST_PATH = paths.SPLIT_MANIFEST_PATH

DEFAULT_TARGET = "no2_pphm"


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def get_feature_columns(grain: str, feature_set: str = "exogenous") -> list[str]:
    manifest = load_manifest()[grain]
    if feature_set == "exogenous":
        features = manifest["exogenous_features"]
    elif feature_set == "autoregressive":
        features = manifest["autoregressive_features"]
    elif feature_set == "all":
        features = manifest["all_candidate_features"]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")
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
    accepts_weight = "sample_weight" in inspect.signature(model.fit).parameters
    if accepts_weight:
        model.fit(X, y, sample_weight=sample_weight)
        return True
    model.fit(X, y)
    return False


def load_station_ids(grain: str, split: str) -> pd.Series:
    path = SPLITS_DIR / f"{grain}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python src/05_train_test_split.py --grain {grain}` first."
        )
    return pd.read_csv(path, usecols=["station_id"], dtype={"station_id": str})["station_id"]


MIN_STATION_ROWS = 30

def evaluate_per_station(y_true, y_pred, station_ids: pd.Series) -> pd.DataFrame:
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
            "station": station_label(station_id),
            "n_rows": n,
            "rmse": float(np.sqrt(mean_squared_error(group["y_true"], group["y_pred"]))),
            "mae": float(mean_absolute_error(group["y_true"], group["y_pred"])),
            "r2": float(r2_score(group["y_true"], group["y_pred"])) if n >= MIN_STATION_ROWS else float("nan"),
        })
    result = pd.DataFrame(rows).sort_values("r2", na_position="first").reset_index(drop=True)
    _remember_predictions(result, frame)
    return result


# --- row-by-row predictions --------------------------------------------------
_PREDICTION_CACHE: dict[int, pd.DataFrame] = {}
_PREDICTION_CACHE_MAX = 16 
_prediction_counter = 0


def _remember_predictions(per_station: pd.DataFrame, frame: pd.DataFrame) -> None:
    global _prediction_counter
    _prediction_counter += 1
    frame = frame.copy()
    frame.insert(0, "row", np.arange(len(frame)))
    _PREDICTION_CACHE[_prediction_counter] = frame
    per_station.attrs["predictions_id"] = _prediction_counter
    while len(_PREDICTION_CACHE) > _PREDICTION_CACHE_MAX:
        _PREDICTION_CACHE.pop(next(iter(_PREDICTION_CACHE)))


def save_per_station(
    model_name: str, grain: str, feature_set: str, split: str, per_station: pd.DataFrame,
    stage: str = "default",
) -> Path:
    out_dir = paths.PER_STATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{model_name}_{stage}_{grain}_{feature_set}_{split}"
    path = out_dir / f"{name}.csv"
    per_station.to_csv(path, index=False)

    predictions = _PREDICTION_CACHE.pop(per_station.attrs.get("predictions_id"), None)
    if predictions is not None and split != "train":
        paths.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(paths.PREDICTIONS_DIR / f"{name}.csv.gz", index=False)
    return path


def evaluate(y_true, y_pred, train_seconds: float | None = None) -> dict:
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if train_seconds is not None:
        metrics["train_seconds"] = round(train_seconds, 3)
    return metrics


def _row_stage(row: dict) -> str:
    stage = row.get("stage") or ("tuned" if row.get("tuned") else "default")
    return "default" if stage == "baseline" else stage


@contextlib.contextmanager
def _results_lock(timeout: float = 120.0):
    lock_path = paths.RESULTS_PATH.with_suffix(".json.lock")
    start = time.time()
    while True:
        try:
            handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(handle)
            break
        except FileExistsError:
            if time.time() - start > timeout:
                lock_path.unlink(missing_ok=True)
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def save_result(
    model_name: str,
    grain: str,
    feature_set: str,
    metrics: dict,
    split: str = "val",
    stage: str = "default",
) -> Path:
    paths.RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_path = paths.RESULTS_PATH
    row = {
        "model": model_name,
        "stage": stage,
        "grain": grain,
        "feature_set": feature_set,
        "split": split,
        **metrics,
    }
    run_id = os.environ.get("PIPELINE_RUN_ID")
    if run_id:
        row["run_id"] = run_id
    with _results_lock():
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
        results.append(row)
        temp_path = results_path.with_suffix(f".json.tmp{os.getpid()}")
        temp_path.write_text(json.dumps(results, indent=2, default=str))
        for attempt in range(40):
            try:
                os.replace(temp_path, results_path)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.25)
    return results_path


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start