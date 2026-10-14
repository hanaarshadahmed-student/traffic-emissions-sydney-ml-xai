"""
Experiment helpers: shared code for the extra experiments run after the main pipeline (leave-one-station-out, ablation, permutation importance). 
Builds and scores models the same way as 06_run_models.py, without touching results.json.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts import paths
from scripts.model_utils import evaluate, fit_with_optional_weight, load_manifest
from models.models import REGISTRY

TARGET = "no2_pphm"
TABULAR_MODELS = ["ridge", "decision_tree", "random_forest", "xgboost", "svr"]


def load_config(path: Path = paths.CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build(name: str, config: dict, n_jobs: int | None = None):
    if name not in TABULAR_MODELS:
        raise SystemExit(f"{name!r}: only the row-based models are supported here "
                         f"({', '.join(TABULAR_MODELS)}) -- LSTM/GRU need sequences and take far longer.")
    params = dict(((config.get("training", {}).get("models", {}).get(name) or {})
                   .get("params") or {}))
    if n_jobs is not None and "n_jobs" in params:
        params["n_jobs"] = n_jobs
    return REGISTRY[name](**params)


def read_split(grain: str, split: str) -> pd.DataFrame:
    path = paths.SPLITS_DIR / f"{grain}_{split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the pipeline first.")
    return pd.read_csv(path, dtype={"station_id": str}, low_memory=False)


def weight_column(grain: str) -> str:
    return load_manifest()[grain]["sample_weight_column"][0]


def fit_score(model, train: pd.DataFrame, evals: dict[str, pd.DataFrame], features: list[str],
              weight_col: str | None) -> dict[str, dict]:
    weights = train[weight_col] if weight_col else None
    if weights is None:
        model.fit(train[features], train[TARGET])
    else:
        fit_with_optional_weight(model, train[features], train[TARGET], weights)
    return {name: evaluate(frame[TARGET], model.predict(frame[features]))
            for name, frame in evals.items() if len(frame)}


def cpu_threads() -> int:
    return os.cpu_count() or 1


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - np.sum((np.asarray(y_pred, float) - y_true) ** 2) / ss_tot) if ss_tot else float("nan")


def explained_xgboost(grain: str, feature_set: str, config: dict, features: list[str]):
    import joblib

    path = paths.SAVED_MODELS_DIR / f"xgboost_{grain}_{feature_set}.joblib"
    if path.exists():
        model = joblib.load(path)
        if list(getattr(model, "feature_names_in_", [])) == list(features):
            return model, f"tuned model ({path.name})"
    train = read_split(grain, "train")
    model = build("xgboost", config)
    fit_with_optional_weight(model, train[features], train[TARGET], train[weight_column(grain)])
    return model, "default XGBoost refitted on train (no matching tuned model saved)"