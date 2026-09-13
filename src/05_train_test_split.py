"""
CO2/NO2 Traffic-Emissions Capstone -- Train/Val/Test Split + Scaling

Splits features_daily.csv / features_hourly.csv into train/val/test and
fits feature scaling -- the step between feature engineering and modeling.

Two choices here matter more than they look:

1. The split is CHRONOLOGICAL, not random, and done PER STATION. Random
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

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
MANIFEST_PATH = PROCESSED_DIR / "feature_manifest.json"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets the remainder (~0.15)

GRAIN_CONFIG = {
    "daily": {
        "input": PROCESSED_DIR / "features_daily.csv",
        "date_col": "date",
    },
    "hourly": {
        "input": PROCESSED_DIR / "features_hourly.csv",
        "date_col": "timestamp",
    },
}


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def assign_chronological_split(df: pd.DataFrame, date_col: str) -> pd.Series:
    """Per-station chronological split. Each station's own rows are sorted
    by time and cut at the train/val boundary and the val/test boundary
    independently -- a station observed for 6 months and one observed for
    2 years both end up with ~70/15/15 of their own timeline in each
    split, rather than one global date cutoff which would leave short-
    history stations entirely in one split."""
    split = pd.Series(index=df.index, dtype=object)
    for station_id, group in df.sort_values(date_col).groupby("station_id"):
        n = len(group)
        train_end = int(np.floor(n * TRAIN_FRAC))
        val_end = train_end + int(np.floor(n * VAL_FRAC))
        labels = np.full(n, "test", dtype=object)
        labels[:train_end] = "train"
        labels[train_end:val_end] = "val"
        split.loc[group.index] = labels
    return split


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


def split_and_scale(grain: str, manifest: dict) -> dict:
    config = GRAIN_CONFIG[grain]
    date_col = config["date_col"]
    df = pd.read_csv(config["input"], dtype={"station_id": str}, low_memory=False)
    df[date_col] = pd.to_datetime(df[date_col])

    df["split"] = assign_chronological_split(df, date_col)

    grain_manifest = manifest[grain]
    candidate_features = grain_manifest["all_candidate_features"]
    scale_cols = pick_scale_columns(df, candidate_features)
    unscaled_cols = [c for c in candidate_features if c not in scale_cols]

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

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
    args = parser.parse_args()

    manifest = load_manifest()
    grains = ["daily", "hourly"] if args.grain == "all" else [args.grain]

    split_manifest = {}
    for grain in grains:
        summary = split_and_scale(grain, manifest)
        split_manifest[grain] = summary
        train, val, test = summary["train"], summary["val"], summary["test"]
        print(
            f"{grain}: train {train['rows']:,} rows ({train['date_min']} to {train['date_max']}), "
            f"val {val['rows']:,} rows ({val['date_min']} to {val['date_max']}), "
            f"test {test['rows']:,} rows ({test['date_min']} to {test['date_max']}) "
            f"-> {grain}_train.csv / {grain}_val.csv / {grain}_test.csv"
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
