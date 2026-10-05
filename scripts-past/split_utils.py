"""
CO2/NO2 Traffic-Emissions Capstone -- Shared chronological split logic

Both 03_data_preprocessing.py and 05_train_test_split.py need to know
which rows belong to "train", and they MUST agree:

- 05 uses it to make the actual train/val/test files and to fit the
  StandardScaler on train only.
- 03 uses it earlier to fit imputation medians on train-only rows.

Importing one function in both places means there's exactly one
definition of "train" to keep in sync.

WHY A SINGLE GLOBAL CUTOFF DATE (not a per-station split)
---------------------------------------------------------
Several traffic stations are matched to the SAME AQ monitoring site, so
they share an identical NO2 target series:

    NEWCASTLE       7212, 7211
    WOLLONGONG      7216, 6178-PR
    GOULBURN        6109, 6135-PR
    PORT MACQUARIE  6124, 6119-PR

The earlier per-station 70/15/15 split cut each station's own timeline
separately. Short stations (7211, 6135-PR, 6178-PR, all 2024-only) had
val/test dates that fell inside their sibling station's TRAIN period --
so the model had already seen the exact target value for that site and
date during training (78-100% of those stations' val/test rows).

With one global cutoff, a calendar date is in exactly one split for
every station, so that can't happen. Consequences, by design:
  - stations whose data ends before the cutoff are entirely in train
  - val/test measure "a later period the model never saw", which is the
    honest question for a temporal forecast
Cutoffs are the 70% / 85% quantiles of all rows' timestamps, floored to
a whole day so small row-count differences between 03 and 05 can't
move the boundary.
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