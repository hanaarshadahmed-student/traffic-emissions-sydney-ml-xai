"""
Split utils: one shared definition of the chronological train/val/test cutoffs (70% / 85% of all timestamps), used by both 03_data_preprocessing.py and 05_train_test_split.py so they always agree on what "train" is.

A single global cutoff date is used (not a per-station split) because some stations share the same NO2 monitor; a per-station split leaked their val/test targets into train.
"""

from __future__ import annotations

import pandas as pd

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets the remainder (~0.15)

def _timeline(df: pd.DataFrame, time_cols) -> pd.Series:
    """One timestamp per row from either a single column ("date",
    "timestamp") or ["date", "hour_ending"] for hourly data."""
    if isinstance(time_cols, str):
        time_cols = [time_cols]
    base = pd.to_datetime(df[time_cols[0]])
    if len(time_cols) > 1 and time_cols[1] == "hour_ending":
        base = base + pd.to_timedelta(df["hour_ending"], unit="h")
    return base


def split_cutoffs(df: pd.DataFrame, time_cols) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(train_end, val_end) -- train is < train_end, val is
    [train_end, val_end), test is >= val_end."""
    ts = _timeline(df, time_cols)
    train_end = ts.quantile(TRAIN_FRAC).floor("D")
    val_end = ts.quantile(TRAIN_FRAC + VAL_FRAC).floor("D")
    return train_end, val_end


def assign_chronological_split(df: pd.DataFrame, time_cols) -> pd.Series:
    """Global-cutoff chronological split -- see module docstring."""
    ts = _timeline(df, time_cols)
    train_end, val_end = split_cutoffs(df, time_cols)
    split = pd.Series("test", index=df.index, dtype=object)
    split[ts < val_end] = "val"
    split[ts < train_end] = "train"
    return split


def train_only_median(df: pd.DataFrame, column: str, time_cols) -> float:
    """Median of `column` over TRAIN rows only, so an imputed value never
    carries information from a row that ends up in val/test."""
    split = assign_chronological_split(df, time_cols)
    return df.loc[split.eq("train"), column].median()