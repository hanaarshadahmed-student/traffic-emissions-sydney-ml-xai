"""
CO2/NO2 Traffic-Emissions Capstone -- Train/Val/Test Split + Scaling

Splits features_daily.csv / features_hourly.csv into train/val/test and
fits feature scaling -- the step between feature engineering and modeling.

Two choices here matter more than they look:

1. The split is CHRONOLOGICAL, not random, with ONE GLOBAL CUTOFF DATE
   for every station (see split_utils.py for why it is no longer per
   station -- stations sharing an AQ site leaked targets across splits). Random
   splitting would leak information: a test-set row's lag/rolling features
   are computed from timestamps that could land in the training set,
   letting the model implicitly see the future. Splitting per station
   (rather than by a single global cutoff date) keeps every station
   represented in train/val/test even though stations have different
   date ranges (see doc/Data.md).

2. The scaler is fit on the TRAIN split ONLY, then applied unchanged to
   val/test. Fitting on the full dataset first would leak test-set
   statistics (mean/std) into training. Binary/one-hot columns (season,
   road bucket, station dummies, availability flags, ...) are left
   unscaled -- scaling a 0/1 flag doesn't help any of the baseline models
   and would make the coefficients harder to read for the linear/SVR
   models downstream.

Input:  data/processed/features_daily.csv
        data/processed/features_hourly.csv
        data/processed/feature_manifest.json
Output: data/processed/splits/{grain}_train.csv
        data/processed/splits/{grain}_val.csv
        data/processed/splits/{grain}_test.csv
        data/processed/splits/{grain}_scaler.joblib
        data/processed/splits/split_manifest.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model_fixed.split_utils import assign_chronological_split as _shared_assign_split
from model_fixed.split_utils import split_cutoffs

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
MANIFEST_PATH = PROCESSED_DIR / "feature_manifest.json"

GRAIN_CONFIG = {
    "daily": {
        "input": PROCESSED_DIR / "features_daily.csv",
        "date_col": "date",
        # previous observed NO2 -> the persistence baseline
        "persistence_source": "no2_pphm_lag_1d",
    },
    "hourly": {
        "input": PROCESSED_DIR / "features_hourly.csv",
        "date_col": "timestamp",
        "persistence_source": "no2_pphm_lag_1h",
    },
}

# Feature pruning: within the exogenous set and within the autoregressive
# set, drop a feature if it's correlated above this |r| (on TRAIN rows
# only) with a feature already kept. Near-duplicates (traffic, log
# traffic, sqrt traffic, ...) add nothing for the models and split SHAP
# credit between them, which makes the XAI step misleading.
PRUNE_THRESHOLD = 0.95

# Name fragments that mark a feature as a derived variant of something
# simpler. When two features are near-duplicates, the one with fewer of
# these is kept -- so the raw, readable feature wins over its transforms.
DERIVED_TOKENS = (
    "log1p", "sqrt", "squared", "_x_", "ratio", "difference", "zscore",
    "vs_prior", "proxy", "change", "share", "imbalance", "_available",
)


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def assign_chronological_split(df: pd.DataFrame, date_col: str) -> pd.Series:
    """Thin wrapper around split_utils' shared implementation -- kept here
    under the same name so nothing else in this file has to change.
    Defined once in split_utils so 03_data_preprocessing.py's train-only
    imputation and this script's train/val/test split can never drift
    apart into disagreeing about which rows are "train"."""
    return _shared_assign_split(df, date_col)


def is_binary_column(series: pd.Series) -> bool:
    """True if a numeric column only ever takes values in {0, 1} (one-hot
    dummies, boolean flags, availability indicators) -- these are left
    unscaled."""
    values = pd.unique(series.dropna())
    return set(np.round(values, 6)).issubset({0.0, 1.0})


def pick_scale_columns(df: pd.DataFrame, candidate_features: list[str]) -> list[str]:
    scale_cols = []
    for col in candidate_features:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            continue
        if is_binary_column(df[col]):
            continue
        scale_cols.append(col)
    return scale_cols


def _derived_score(name: str) -> int:
    return sum(token in name for token in DERIVED_TOKENS)


def prune_features(train_df: pd.DataFrame, features: list[str]) -> dict[str, str]:
    """Returns {dropped_feature: reason}. Fit on TRAIN rows only."""
    present = [f for f in features if f in train_df.columns]
    dropped: dict[str, str] = {}
    variable = []
    for f in present:
        if train_df[f].nunique(dropna=True) <= 1:
            dropped[f] = "constant in train"
        else:
            variable.append(f)
    corr = train_df[variable].astype(float).corr().abs()
    order = sorted(variable, key=lambda f: (_derived_score(f), variable.index(f)))
    kept: list[str] = []
    for f in order:
        twin = next((k for k in kept if corr.at[f, k] > PRUNE_THRESHOLD), None)
        if twin is None:
            kept.append(f)
        else:
            dropped[f] = f"|r|={corr.at[f, twin]:.3f} with kept '{twin}'"
    return dropped


def split_and_scale(grain: str, manifest: dict, prune: bool = True) -> dict:
    config = GRAIN_CONFIG[grain]
    date_col = config["date_col"]
    df = pd.read_csv(config["input"], dtype={"station_id": str}, low_memory=False)
    df[date_col] = pd.to_datetime(df[date_col])

    df["split"] = assign_chronological_split(df, date_col)
    train_end, val_end = split_cutoffs(df, date_col)

    # Unscaled copy of yesterday's / last hour's NO2 for the persistence
    # baseline in 06_run_models.py. Not in the feature manifest, so no
    # model ever trains on it.
    persistence_col = config["persistence_source"]
    if persistence_col in df.columns:
        df["baseline_persistence"] = df[persistence_col]
        available = f"{persistence_col}_available"
        if available in df.columns:
            df.loc[df[available] == 0, "baseline_persistence"] = np.nan

    grain_manifest = manifest[grain]
    candidate_features = grain_manifest["all_candidate_features"]
    scale_cols = pick_scale_columns(df, candidate_features)
    unscaled_cols = [c for c in candidate_features if c not in scale_cols]

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    excluded: dict[str, str] = {}
    if prune:
        for group in ("exogenous_features", "autoregressive_features"):
            excluded.update(prune_features(train_df, grain_manifest[group]))

    scaler = StandardScaler()
    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    val_df[scale_cols] = scaler.transform(val_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / f"{grain}_train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / f"{grain}_val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / f"{grain}_test.csv", index=False)
    joblib.dump(scaler, SPLITS_DIR / f"{grain}_scaler.joblib")

    def split_summary(part_df: pd.DataFrame) -> dict:
        return {
            "rows": int(len(part_df)),
            "stations": int(part_df["station_id"].nunique()),
            "date_min": str(part_df[date_col].min()),
            "date_max": str(part_df[date_col].max()),
        }

    return {
        "split_method": "global chronological cutoff (see split_utils.py)",
        "train_end_exclusive": str(train_end),
        "val_end_exclusive": str(val_end),
        "excluded_features": excluded,
        "train": split_summary(train_df),
        "val": split_summary(val_df),
        "test": split_summary(test_df),
        "scaled_features": scale_cols,
        "unscaled_features": unscaled_cols,
        "target_columns": grain_manifest["target_columns"],
        "sample_weight_column": grain_manifest["sample_weight_column"],
        "exogenous_features": grain_manifest["exogenous_features"],
        "autoregressive_features": grain_manifest["autoregressive_features"],
        "all_candidate_features": candidate_features,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grain", choices=["daily", "hourly", "all"], default="all")
    parser.add_argument("--no-prune", action="store_true",
                        help="Keep every manifest feature (skip near-duplicate pruning).")
    args = parser.parse_args()

    manifest = load_manifest()
    grains = ["daily", "hourly"] if args.grain == "all" else [args.grain]

    split_manifest = {}
    for grain in grains:
        summary = split_and_scale(grain, manifest, prune=not args.no_prune)
        split_manifest[grain] = summary
        train, val, test = summary["train"], summary["val"], summary["test"]
        print(
            f"{grain}: train {train['rows']:,} rows ({train['date_min']} to {train['date_max']}), "
            f"val {val['rows']:,} rows ({val['date_min']} to {val['date_max']}), "
            f"test {test['rows']:,} rows ({test['date_min']} to {test['date_max']}) "
            f"-> {grain}_train.csv / {grain}_val.csv / {grain}_test.csv"
        )
        print(
            f"       cutoffs: train < {summary['train_end_exclusive'][:10]} <= val < "
            f"{summary['val_end_exclusive'][:10]} <= test"
        )
        n_ex = len(summary["exogenous_features"]) - sum(f in summary["excluded_features"] for f in summary["exogenous_features"])
        n_ar = len(summary["autoregressive_features"]) - sum(f in summary["excluded_features"] for f in summary["autoregressive_features"])
        print(
            f"       pruned {len(summary['excluded_features'])} near-duplicate/constant features "
            f"-> {n_ex} exogenous + {n_ar} autoregressive kept (list in split_manifest.json)"
        )
        print(
            f"       {len(summary['scaled_features'])} scaled features, "
            f"{len(summary['unscaled_features'])} left unscaled (binary/one-hot)"
        )

    with open(SPLITS_DIR / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"split manifest -> {SPLITS_DIR / 'split_manifest.json'}")


if __name__ == "__main__":
    main()